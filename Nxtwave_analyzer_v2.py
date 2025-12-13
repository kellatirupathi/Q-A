import datetime
import streamlit as st
import pandas as pd
import sys
import os
import requests
import json
import gdown
import gspread
import subprocess
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from io import StringIO

# ==============================================================================
#  STREAMLIT PAGE CONFIGURATION (FULL WIDTH)
# ==============================================================================
st.set_page_config(page_title="Interview Analyzer", layout="wide")

# Load environment variables
load_dotenv()

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_TRANSCRIBE_API_KEY = os.getenv("MISTRAL_TRANSCRIBE_API_KEY") or MISTRAL_API_KEY

# Endpoints
MISTRAL_TRANSCRIBE_URL = "https://api.mistral.ai/v1/audio/transcriptions"
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-large-latest"

# Sheets Configuration
GSHEET_NAME = "Google Drive Video to Q&A Analyzer"
NXTWAVE_QA_SUBSHEET_NAME = "Nxtwave Q&A"

# ==============================================================================
#  STRICT PROMPT (Logic Improved for Consistency)
# ==============================================================================
QNA_PROMPT_STRICT = """
#### Task: Technical Interview Extraction & Classification ####

**Role:** Senior Technical Interview Auditor
**Goal:** Extract questions/answers and strictly categorize the "Techstack name" based on the technology discussed, NOT the context of where it was learned.

**CRITICAL RULES FOR TECH STACK LABELING:**
1. **OOP is Technical:** Questions about Classes, Objects, Inheritance, Polymorphism, or Encapsulation must be labeled as "**OOP**" or the specific language (e.g., "**Java**", "**PHP**", "**Python**"). **NEVER** label these as "General Aptitude".
2. **Project vs. Tech:** 
   - If the question is "What is Laravel?", label it "**PHP**". (Do NOT label it "Project").
   - If the question is "Explain your project architecture", label it "**Project**".
   - If the question is "How did you use Authentication in your project?", label it "**Security**" or "**PHP**".
3. **Web Development:** Do NOT use the term "Web Development". 
   - If it's about tags/DOM, use "**HTML**". 
   - If it's about styling, use "**CSS**".
   - If it's about backend logic, use "**Node.js**", "**PHP**", etc.

**Allowed Techstack Names:**
["Python", "Java", "C++", "JavaScript", "React", "Node.js", "HTML", "CSS", "SQL", "MongoDB", "PHP", "OOP", "DSA", "System Design", "General Aptitude", "HR", "Project"]

**Output Format (JSON):**
[
  {
    "question_text": "Exact question asked",
    "answer_text": "Summarized answer given by candidate",
    "question_type": "Theory" or "Coding" or "Behavioral",
    "question_concept": "One of the Allowed Techstack Names",
    "difficulty": "Easy" or "Medium" or "Hard"
  }
]
"""

# ==============================================================================
#  GCP AUTHENTICATION
# ==============================================================================

def get_gcp_credentials_dict():
    return {
        "type": os.getenv("GCP_TYPE"),
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GCP_PRIVATE_KEY").replace('\\n', '\n') if os.getenv("GCP_PRIVATE_KEY") else None,
        "client_email": os.getenv("GCP_CLIENT_EMAIL"),
        "client_id": os.getenv("GCP_CLIENT_ID"),
        "auth_uri": os.getenv("GCP_AUTH_URI"),
        "token_uri": os.getenv("GCP_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GCP_CLIENT_CERT_URL")
    }

def get_gspread_client():
    try:
        creds_dict = get_gcp_credentials_dict()
        if not creds_dict["private_key"]: return None
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except: return None

gs_client = get_gspread_client()

# ==============================================================================
#  FFMPEG MEDIA PROCESSING
# ==============================================================================

def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def get_media_duration(media_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", media_path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
        return float(result.stdout)
    except: return None

def extract_audio_segment(video_path, audio_output_path, start_time, end_time):
    command = [
        "ffmpeg", "-i", video_path, "-ss", str(start_time), "-to", str(end_time),
        "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_output_path, "-y"
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_audio_into_chunks(audio_path, chunk_duration_seconds=600):
    total_duration = get_media_duration(audio_path)
    if total_duration is None or total_duration <= chunk_duration_seconds:
        return [audio_path]
    
    audio_dir = os.path.dirname(audio_path)
    audio_base = os.path.splitext(os.path.basename(audio_path))[0]
    chunk_paths = []
    
    start = 0
    i = 0
    while start < total_duration:
        chunk_name = os.path.join(audio_dir, f"{audio_base}_part_{i}.mp3")
        command = [
            "ffmpeg", "-i", audio_path, "-ss", str(start), "-t", str(chunk_duration_seconds),
            "-c", "copy", chunk_name, "-y"
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunk_paths.append(chunk_name)
        start += chunk_duration_seconds
        i += 1
    
    return chunk_paths

def format_timestamp(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = 0
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def mistral_json_to_srt(segments, offset_seconds=0):
    srt_output = ""
    for i, segment in enumerate(segments):
        start = segment.get('start', 0) + offset_seconds
        end = segment.get('end', 0) + offset_seconds
        text = segment.get('text', '').strip()
        srt_output += f"{i + 1}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n"
    return srt_output

# ==============================================================================
#  MISTRAL API FUNCTIONS
# ==============================================================================

def get_mistral_completion(prompt, content):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt + " Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": content}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(MISTRAL_CHAT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        res_json = response.json()
        clean_text = res_json['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
        return clean_text
    except Exception as e:
        print(f"Mistral Chat Error: {e}")
        return "[]"

def generate_transcript(audio_path, output_srt_path):
    chunks = split_audio_into_chunks(audio_path)
    
    headers = {"Authorization": f"Bearer {MISTRAL_TRANSCRIBE_API_KEY}"}
    data = {"model": "voxtral-mini-latest", "timestamp_granularities": ["segment"], "response_format": "verbose_json"}
    
    total_offset = 0
    
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            with open(chunk, "rb") as audio_file:
                files = {"file": (os.path.basename(chunk), audio_file, "audio/mpeg")}
                try:
                    resp = requests.post(MISTRAL_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=300)
                    if resp.status_code == 200:
                        segments = resp.json().get("segments", [])
                        f.write(mistral_json_to_srt(segments, total_offset))
                        total_offset += get_media_duration(chunk)
                except Exception as e:
                    print(f"Transcribe Error: {e}")
            if chunk != audio_path:
                os.remove(chunk)

# ==============================================================================
#  DATA PROCESSING & NORMALIZATION
# ==============================================================================

def load_file_content(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def process_transcript_in_chunks(transcript, prompt, chunk_size=4000):
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    final_data = []
    prev_context = ""
    
    for chunk in chunks:
        combined_input = f"Previous Context: {prev_context}\n\nCurrent Text: {chunk}"
        result = get_mistral_completion(prompt, combined_input)
        try:
            data = json.loads(result)
            if isinstance(data, list): final_data.extend(data)
            elif isinstance(data, dict): final_data.append(data)
            prev_context = str(data) 
        except: pass
    
    return final_data

def normalize_tech_stack(concept):
    """
    Enforces Strict Naming Conventions.
    Fixes common AI mapping errors (e.g., 'Laravel' -> 'PHP', 'OOPS' -> 'OOP', 'Web Dev' -> 'HTML')
    """
    if not isinstance(concept, str): return "Other"
    
    c = concept.strip().lower()
    
    # DICTIONARY LOOKUP FOR PRECISION
    mapping = {
        "laravel": "PHP",
        "php framework": "PHP",
        "oops": "OOP",
        "object oriented programming": "OOP",
        "object-oriented programming": "OOP",
        "web development": "HTML",
        "web dev": "HTML",
        "reactjs": "React",
        "react.js": "React",
        "nodejs": "Node.js",
        "express": "Node.js",
        "express.js": "Node.js",
        "mongo": "MongoDB",
        "mysql": "SQL",
        "postgres": "SQL",
        "postgresql": "SQL",
        "dbms": "SQL",
        "database": "SQL",
        "cpp": "C++",
        "java script": "JavaScript",
        "js": "JavaScript",
        "aptitude": "General Aptitude",
        "reasoning": "General Aptitude",
        "behavioral": "HR",
        "introduction": "HR"
    }
    
    # 1. Direct Match
    if c in mapping:
        return mapping[c]
    
    # 2. Substring Matching (Order Matters)
    if 'react' in c: return 'React'
    if 'node' in c: return 'Node.js'
    if 'python' in c: return 'Python'
    if 'java' in c and 'script' not in c: return 'Java'
    if 'javascript' in c: return 'JavaScript'
    if 'sql' in c: return 'SQL'
    if 'html' in c: return 'HTML'
    if 'css' in c: return 'CSS'
    if 'php' in c: return 'PHP'
    if 'aws' in c or 'cloud' in c: return 'AWS'
    if 'project' in c: return 'Project'
    
    # 3. Capitalize if unknown
    return concept.title()

def write_to_nxtwave_sheet_final(data_df):
    """
    Writes the specific single consolidated table to Google Sheets.
    """
    if gs_client is None: 
        st.error("Google Sheets Client not initialized.")
        return
    try:
        sh = gs_client.open(GSHEET_NAME)
        try:
            worksheet = sh.worksheet(NXTWAVE_QA_SUBSHEET_NAME)
        except gspread.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=NXTWAVE_QA_SUBSHEET_NAME, rows=1000, cols=20)
        
        headers = [
            "user_id", "job_id", "Questions", "Answers", "Question type", 
            "Techstack name", "Difficulty", "interview_round", 
            "clip_start_time", "clip_end_time", "drive_file_id"
        ]
        
        current_headers = worksheet.row_values(1)
        if not current_headers:
            worksheet.append_row(headers)
        
        rows = data_df[headers].values.tolist()
        
        if rows: 
            worksheet.append_rows(rows)
            
    except Exception as e: st.error(f"Sheet Error: {e}")

# ==============================================================================
#  MAIN APP UI
# ==============================================================================

DIRS = ['DownloadedVideos', 'GeneratedTranscripts', 'Q&A']
for d in DIRS:
    if not os.path.exists(d): os.makedirs(d)

if "button1" not in st.session_state: st.session_state["button1"] = False

input_df = pd.DataFrame()
is_input_valid = False
REQUIRED_COLUMNS = ['drive_file_id', 'user_id', 'job_id', 'interview_round', 'clip_start_time', 'clip_end_time']

input_method = st.radio("Choose input method:", ["Upload CSV", "Paste Text"], horizontal=True, index=1)

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "txt"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if all(col in input_df.columns for col in REQUIRED_COLUMNS): is_input_valid = True
            else: st.error(f"Missing columns: {REQUIRED_COLUMNS}")
        except: st.error("Invalid CSV")

elif input_method == "Paste Text":
    paste_text = st.text_area("Paste Data (Tab-Separated)", height=150,
        placeholder="1gfZvlSqXO0RnXz...\trahul_123\tJ_101\t1\t0\t300",
        help="drive_file_id, user_id, job_id, interview_round, start_time, end_time")
    if paste_text.strip():
        try:
            input_df = pd.read_csv(StringIO(paste_text), sep='\t', header=None, names=REQUIRED_COLUMNS)
            is_input_valid = True
        except: st.error("Invalid Text Format")

if is_input_valid and not input_df.empty:
    st.dataframe(input_df, use_container_width=True)

# ==============================================================================
#  PROCESSING PIPELINE
# ==============================================================================

def process_single_row(row):
    file_id = str(row['drive_file_id']).strip()
    user_id = str(row['user_id'])
    job_id = str(row['job_id'])
    round_no = str(row['interview_round'])
    start = int(row['clip_start_time'])
    end = int(row['clip_end_time'])
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Paths
    base_name = f"{file_id}_{start}_{end}"
    video_path = os.path.join("DownloadedVideos", f"{file_id}.mp4")
    audio_path = os.path.join("DownloadedVideos", f"{base_name}.mp3")
    transcript_path = os.path.join("GeneratedTranscripts", f"{base_name}.srt")
    qna_path = os.path.join("Q&A", f"{base_name}_consolidated.csv")

    st.markdown("---")
    with st.expander(f"🚀 Processing: {user_id} | Job: {job_id}", expanded=True):
        st.write(f"**Drive ID:** `{file_id}` | **Time:** `{start}-{end}` seconds")

        # 1. Download Video
        if not os.path.exists(video_path):
            with st.spinner("📥 Downloading Video..."):
                gdown.download(url, video_path, quiet=False, fuzzy=True)
        
        if not os.path.exists(video_path):
            st.error("❌ Failed to download video")
            return
        else:
            st.success("✅ Video Downloaded")
            st.video(video_path, start_time=start)

        # 2. Extract Audio
        if not os.path.exists(audio_path):
            with st.spinner("🎧 Extracting Audio..."):
                extract_audio_segment(video_path, audio_path, start, end)
        
        # 3. Transcribe
        if not os.path.exists(transcript_path):
            with st.spinner("🗣️ Generating Transcript..."):
                generate_transcript(audio_path, transcript_path)
        
        transcript_text = load_file_content(transcript_path)
        
        # 4. Generate Consolidated Table
        st.info("🧠 Analyzing & Consolidating Data...")
        if not os.path.exists(qna_path):
            with st.spinner("Extracting Q&A, Tech Stack, Difficulty..."):
                qna_json = process_transcript_in_chunks(transcript_text, QNA_PROMPT_STRICT)
                qna_df = pd.DataFrame(qna_json)
                
                if not qna_df.empty:
                    # Rename columns to match desired output
                    qna_df.rename(columns={
                        'question_text': 'Questions',
                        'answer_text': 'Answers',
                        'question_type': 'Question type',
                        'question_concept': 'Techstack name',
                        'difficulty': 'Difficulty'
                    }, inplace=True)

                    # Add Metadata columns
                    qna_df['user_id'] = user_id
                    qna_df['job_id'] = job_id
                    qna_df['interview_round'] = round_no
                    qna_df['clip_start_time'] = start
                    qna_df['clip_end_time'] = end
                    qna_df['drive_file_id'] = file_id
                    
                    # Apply STRICT Tech Stack Normalization
                    qna_df['Techstack name'] = qna_df['Techstack name'].apply(normalize_tech_stack)
                    
                    if 'Difficulty' not in qna_df.columns:
                        qna_df['Difficulty'] = 'Medium' 
                    
                    qna_df.to_csv(qna_path, index=False)
                else:
                    st.warning("No Q&A pairs found.")
        
        # 5. Display & Auto-Save
        if os.path.exists(qna_path):
            final_df = pd.read_csv(qna_path)
            
            display_cols = ["user_id", "job_id", "Questions", "Answers", "Question type", 
                            "Techstack name", "Difficulty", "interview_round", 
                            "clip_start_time", "clip_end_time", "drive_file_id"]
            
            for col in display_cols:
                if col not in final_df.columns:
                    final_df[col] = "N/A"
            
            final_df = final_df[display_cols]
            
            st.subheader("📝 Final Extracted Data")
            edited_final_df = st.data_editor(final_df, key=f"final_{base_name}", use_container_width=True, num_rows="dynamic")

            autosave_key = f"autosaved_{base_name}"
            if autosave_key not in st.session_state:
                with st.spinner(f"☁️ Autosaving results for {user_id} to Google Sheets..."):
                    write_to_nxtwave_sheet_final(edited_final_df)
                    st.session_state[autosave_key] = True
                    st.toast(f"✅ Automatically saved {len(edited_final_df)} rows for {user_id}!", icon="☁️")
            else:
                st.success(f"✅ Data for {user_id} is already saved to Google Sheets.")

if st.button("🚀 Start Analysis Pipeline"):
    if not check_ffmpeg_installed():
        st.error("⚠️ FFmpeg is not installed or not in PATH. Please install it to proceed.")
    elif is_input_valid:
        for idx, row in input_df.iterrows():
            try:
                process_single_row(row)
            except Exception as e:
                st.error(f"Error processing row {idx}: {e}")
        st.success("🎉 All processing complete!")
        st.balloons()
    else:
        st.error("Please provide valid input data via CSV or Paste Text.")