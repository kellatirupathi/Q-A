import streamlit as st
import requests
import subprocess
import os
import re
import gdown
from datetime import datetime
import pandas as pd
import json
import logging
from io import StringIO
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================================================
#  CONFIGURATION & INITIALIZATION
# ==============================================================================

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Secrets & API Keys (Loaded from .env) ---
# We use the same key for both transcribe and chat if they are the same in your .env
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
# Fallback if MISTRAL_API_KEY isn't set but TRANSCRIBE is
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = os.getenv("MISTRAL_TRANSCRIBE_API_KEY")

GITHUB_GIST_TOKEN = os.getenv("GITHUB_GIST_TOKEN")

# Mistral Endpoints
MISTRAL_TRANSCRIBE_URL = "https://api.mistral.ai/v1/audio/transcriptions"
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

# --- Google Sheets Configuration ---
GSHEET_NAME = "Google Drive Video to Q&A Analyzer"
QA_SUBSHEET_NAME = "NIAT Q&A"

# --- GitHub Gist API URL ---
GIST_API_URL = "https://api.github.com/gists"


# ==============================================================================
#  HELPER: Construct Credentials Dict from ENV
# ==============================================================================
def get_gcp_credentials_dict():
    """Builds the dictionary expected by google-auth from .env variables."""
    return {
        "type": os.getenv("GCP_TYPE"),
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
        # Handle newlines in private key correctly
        "private_key": os.getenv("GCP_PRIVATE_KEY").replace('\\n', '\n') if os.getenv("GCP_PRIVATE_KEY") else None,
        "client_email": os.getenv("GCP_CLIENT_EMAIL"),
        "client_id": os.getenv("GCP_CLIENT_ID"),
        "auth_uri": os.getenv("GCP_AUTH_URI"),
        "token_uri": os.getenv("GCP_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GCP_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.getenv("GCP_CLIENT_CERT_URL")
    }

# ==============================================================================
#  GOOGLE SHEETS & GIST INTEGRATION
# ==============================================================================

@st.cache_resource
def get_gspread_client():
    """Initializes and returns a gspread client, caching it for performance."""
    try:
        creds_dict = get_gcp_credentials_dict()
        
        # Check if critical keys exist
        if not creds_dict["private_key"] or not creds_dict["client_email"]:
            logger.error("GCP Credentials not found in .env.")
            st.error("Missing GCP Service Account credentials in .env. Cannot connect to Google Sheets.")
            return None
            
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        
        logger.info("Successfully authorized with Google Sheets API.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {e}")
        st.error(f"Failed to connect to Google Sheets. Check .env and sharing permissions. Error: {e}")
        return None


def get_or_create_worksheet(client, sheet_name, subsheet_name):
    """Gets a subsheet by name, creating it if it doesn't exist."""
    if not client:
        return None
    try:
        spreadsheet = client.open(sheet_name)
        try:
            worksheet = spreadsheet.worksheet(subsheet_name)
            logger.info(f"Found existing worksheet: '{subsheet_name}'.")
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=subsheet_name, rows=1000, cols=30)
            logger.info(f"Created new worksheet: '{subsheet_name}' in '{sheet_name}'.")
        return worksheet
    except gspread.SpreadsheetNotFound:
        st.error(f"Spreadsheet '{sheet_name}' not found. Please create it and share it with the service account email.")
        return None
    except Exception as e:
        st.error(f"An error occurred accessing the spreadsheet: {e}")
        return None


def write_df_to_gsheet(worksheet, df_to_write, video_link, file_id, company_name, uid, niat_id, student_name):
    """Appends a DataFrame's data to the given worksheet, ensuring columns align and updating headers if needed."""
    if not worksheet or df_to_write is None or df_to_write.empty:
        st.warning("No data to write to Google Sheets.")
        return
        
    try:
        logger.info(f"Preparing to write {len(df_to_write)} rows for {company_name} to Google Sheets.")
        
        # Add all metadata columns to the DataFrame
        df_to_write['UID'] = uid
        df_to_write['NIAT ID'] = niat_id
        df_to_write['Name of the student'] = student_name
        df_to_write['Company Name'] = company_name
        df_to_write['Video Link'] = video_link
        df_to_write['File ID'] = file_id
        df_to_write['Analysis Datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # --- DYNAMIC HEADER CHECKING AND UPDATING ---
        required_header = df_to_write.columns.tolist()
        try:
            existing_header = worksheet.row_values(1)
        except gspread.exceptions.APIError:
            existing_header = []

        if not existing_header:
            worksheet.update('A1', [required_header])
            final_header = required_header
            logger.info("Sheet was empty. Created new header.")
        else:
            missing_columns = [col for col in required_header if col not in existing_header]
            if missing_columns:
                logger.warning(f"Header mismatch. Missing columns in GSheet: {missing_columns}. Appending them.")
                final_header = existing_header + missing_columns
                worksheet.update('A1', [final_header])
            else:
                final_header = existing_header
        
        # Now, align the DataFrame using the final, correct header
        aligned_df = df_to_write.reindex(columns=final_header).fillna('')
        rows_to_append = aligned_df.values.tolist()
        
        if rows_to_append:
            worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
            logger.info(f"Successfully wrote {len(rows_to_append)} rows to '{worksheet.title}'.")
            
    except Exception as e:
        logger.error(f"Failed to write to Google Sheets: {e}")
        st.toast(f"⚠️ Could not write to Google Sheet. Error: {e}", icon="📄")


def create_gist(filename, content):
    """Creates an authenticated, public GitHub Gist and returns the public link."""
    try:
        token = GITHUB_GIST_TOKEN
        if not token:
            st.error("`GITHUB_GIST_TOKEN` not found in `.env`. Cannot create transcript link.")
            return None

        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        
        payload = {
            'description': f'Transcript for {filename}',
            'public': True,
            'files': {
                filename: {
                    'content': content
                }
            }
        }
        
        response = requests.post(GIST_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        
        data = response.json()
        public_link = data.get('html_url')
        
        if public_link:
            return public_link
        else:
            st.warning("Gist created, but could not find a public link in the response.")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to create public transcript link via GitHub Gist: {e}")
        if e.response:
            st.json(e.response.json())
        return None


# ==============================================================================
#  FILE & DIRECTORY MANAGEMENT
# ==============================================================================

def setup_persistent_directories(segment_id):
    """Creates a unique directory for a specific video time segment."""
    base_output_dir = os.path.join(os.getcwd(), "processing_output")
    segment_run_dir = os.path.join(base_output_dir, segment_id)
    os.makedirs(segment_run_dir, exist_ok=True)
    st.info(f"📁 Outputs for this segment are in: `{segment_run_dir}`")
    return segment_run_dir


def get_file_id_from_link(link):
    """Extracts the Google Drive file ID from a share link."""
    match = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{10,})", link)
    return match.group(1) if match else None


def save_file(content, directory, filename):
    """Saves text content to a file in a specified directory."""
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
    except IOError as e:
        st.error(f"❌ Failed to save file {filename}: {e}")
        return None


# ==============================================================================
# CORE AUDIO/VIDEO PROCESSING
# ==============================================================================

def check_ffmpeg_installed():
    """Verifies that ffmpeg and ffprobe are available in the system's PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def get_media_duration(media_path):
    """Returns the duration of a media file in seconds."""
    command = [
        "ffprobe", "-v", "error", "-show_entries", 
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", media_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        return float(result.stdout)
    except Exception:
        return None


def format_timestamp(seconds):
    """Converts seconds into HH:MM:SS format."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def parse_time_to_seconds(time_str):
    """Converts HH:MM:SS string to seconds, for timestamp parsing."""
    if not time_str: return 0
    parts = time_str.split(':')
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2].replace(',', '.')) if len(parts) > 2 else 0.0
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0

def clean_transcript(transcript_text, max_consecutive=3, time_window_seconds=5):
    """
    Cleans a transcript by removing rapid, consecutive repetitions of the same short phrase.
    """
    lines = transcript_text.strip().split('\n')
    cleaned_lines = []
    i = 0
    while i < len(lines):
        current_line = lines[i]
        
        match = re.search(r'\[(.*?)\]\s*(.*)', current_line)
        if not match:
            cleaned_lines.append(current_line) # Keep lines that don't match format
            i += 1
            continue
        
        current_ts_str, current_text = match.groups()
        current_text = current_text.strip().lower()
        
        if len(current_text.split()) > 4 or not current_text: 
            cleaned_lines.append(current_line)
            i += 1
            continue

        repeating_block = [current_line]
        lookahead_index = i + 1
        
        while lookahead_index < len(lines):
            next_line = lines[lookahead_index]
            next_match = re.search(r'\[(.*?)\]\s*(.*)', next_line)
            if not next_match:
                break 
            
            next_ts_str, next_text = next_match.groups()
            next_text = next_text.strip().lower()

            time_diff = abs(parse_time_to_seconds(next_ts_str) - parse_time_to_seconds(current_ts_str))

            if next_text == current_text and time_diff < time_window_seconds:
                repeating_block.append(next_line)
                lookahead_index += 1
            else:
                break

        if len(repeating_block) > max_consecutive:
            cleaned_lines.append(repeating_block[0])
            i = lookahead_index 
        else:
            cleaned_lines.extend(repeating_block)
            i = lookahead_index
            
    return "\n".join(cleaned_lines)


def download_drive_video_with_progress(drive_link, file_id):
    """Downloads a video from Google Drive, caching it to avoid re-downloads."""
    video_cache_dir = os.path.join(os.getcwd(), "video_cache")
    os.makedirs(video_cache_dir, exist_ok=True)
    video_path = os.path.join(video_cache_dir, f"{file_id}.mp4")

    if os.path.exists(video_path):
        st.success(f"✅ Video already downloaded (found in cache).")
        return video_path

    url = f"https://drive.google.com/uc?id={file_id}"
    st.write("📥 Downloading Video...")
    progress_bar = st.progress(0)
    progress_status = st.empty()
    try:
        command = ["gdown", "--no-cookies", url, "-O", video_path]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            errors='replace'
        )
        
        full_output = ""
        for line in process.stdout: 
            clean_line = line.strip()
            full_output += line
            
            if '%' in clean_line:
                progress_status.text(clean_line)
            
            if match := re.search(r'(\d+)\s*%', clean_line):
                progress_bar.progress(int(match.group(1)))
        
        process.wait()
        
        if process.returncode != 0 or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            st.error(f"❌ Gdown download failed.")
            st.code(full_output or "No output from gdown, but the download was unsuccessful.")
            if os.path.exists(video_path): os.remove(video_path)
            return None

        progress_bar.progress(100)
        progress_status.success("✅ Video download complete!")
        return video_path
    except Exception as e:
        st.error(f"❌ An error occurred during download: {e}")
        return None

def extract_audio_segment_with_progress(local_video_path, file_id, run_dir, start_time, end_time):
    """Extracts a specific audio segment from a locally stored video file."""
    segment_duration_seconds = end_time - start_time
    audio_filename = f"{file_id}_{start_time}-{end_time}.mp3"
    audio_path = os.path.join(run_dir, audio_filename)
    
    if os.path.exists(audio_path):
        st.success(f"✅ Audio segment already exists.")
        return audio_path

    st.write(f"🎧 Extracting Audio from {format_timestamp(start_time)} to {format_timestamp(end_time)}...")
    progress_bar = st.progress(0)
    progress_status = st.empty()
    
    command = [
        "ffmpeg",
        "-ss", str(start_time),   
        "-i", local_video_path,   
        "-to", str(end_time),     
        "-vn", "-acodec", "libmp3lame", "-q:a", "2",
        audio_path, "-y", "-progress", "pipe:2"
    ]

    try:
        process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, universal_newlines=True, errors='replace')
        ffmpeg_output = ""
        for line in iter(process.stderr.readline, ''):
            ffmpeg_output += line
            if "time=" in line:
                if match := re.search(r"time=(\d{2}:\d{2}:\d{2}[.,]\d+)", line):
                    if segment_duration_seconds > 0:
                        elapsed_seconds = parse_time_to_seconds(match.group(1))
                        percent_complete = int((elapsed_seconds / segment_duration_seconds) * 100)
                        
                        progress_value = min(100, max(0, percent_complete))
                        progress_bar.progress(progress_value)
                        
                        display_elapsed = min(elapsed_seconds, segment_duration_seconds)
                        progress_status.text(f"Processing... {format_timestamp(display_elapsed)} / {format_timestamp(segment_duration_seconds)}")
        process.wait()

        if process.returncode != 0:
            st.error("Audio extraction failed. FFmpeg returned an error.")
            st.code(ffmpeg_output, language="bash") 
            return None
        
        progress_bar.progress(100)
        progress_status.success("✅ Audio segment extracted!")
        return audio_path
        
    except ValueError as e:
        st.error(f"An error occurred during audio extraction: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        return None

def split_audio_into_chunks(audio_path, chunk_duration_seconds=840):
    """Splits a long audio file into smaller chunks for transcription."""
    total_duration = get_media_duration(audio_path)
    if total_duration is None or total_duration <= chunk_duration_seconds:
        return [audio_path]
    audio_dir, audio_base = os.path.dirname(audio_path), os.path.splitext(os.path.basename(audio_path))[0]
    existing_chunks = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.startswith(f"{audio_base}_chunk_")])
    if existing_chunks:
        return existing_chunks
    with st.spinner(f"🔪 Audio is long ({format_timestamp(total_duration)}). Splitting into chunks..."):
        chunk_paths, start_time, i = [], 0, 0
        while start_time < total_duration:
            chunk_path = os.path.join(audio_dir, f"{audio_base}_chunk_{i+1}.mp3")
            command = [
                "ffmpeg", "-i", audio_path, "-ss", str(start_time), 
                "-t", str(chunk_duration_seconds), "-c", "copy", chunk_path, "-y"
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_paths.append(chunk_path)
            start_time += chunk_duration_seconds
            i += 1
    return chunk_paths


def transcribe_audio_chunk(audio_path, session):
    """Transcribes a single audio chunk using the Mistral API with retries."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    data = {"model": "voxtral-mini-latest", "timestamp_granularities": ["segment"]}
    
    with open(audio_path, "rb") as audio_file:
        files = {"file": (os.path.basename(audio_path), audio_file, "audio/mpeg")}
        try:
            response = session.post(MISTRAL_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=600)
            response.raise_for_status() 
            return response.json().get("segments", [])
        except requests.exceptions.RequestException as e:
            st.error(f"API Error on {os.path.basename(audio_path)} after retries: {e}")
            return None


def generate_qa_dataframe(full_transcript_path):
    """Sends the transcript to Mistral (Chat) to extract a structured Q&A DataFrame."""
    st.write(f"Analyzing transcript (`{os.path.basename(full_transcript_path)}`) with Mistral...")
    try:
        with open(full_transcript_path, 'r', encoding='utf-8') as f:
            full_transcript_text = f.read()
    except IOError as e:
        st.error(f"❌ Could not read the full transcript file: {e}")
        return None
        
    if not MISTRAL_API_KEY:
        st.error("MISTRAL_API_KEY not found in .env.")
        return None
        
    # --- UPDATED STRICT PROMPT ---
    prompt = f"""
    You are a Senior Technical Interview Auditor. Your mission is to extract **only strictly technical and practical interview questions** from the provided transcript.

    🧠 **CRITICAL EXTRACTION RULES:**
    
    1.  **TECH STACK EXTRACTION (STRICT):** 
        - Identify the technology based **ONLY on the Question**, not the context of where the student learned it or the answer provided.
        - If the question is "What is a Promise?", the Tech Stack is **"JavaScript"** (even if the student mentions Python).
        - If the question is "Explain Normalization", the Tech Stack is **"SQL"** or **"DBMS"**.
        - **Standardized Naming Conventions:**
            - "React.js" -> **"Reactjs"**
            - "NodeJS" / "Express" -> **"Nodejs"**
            - "Web Development" (tags/DOM) -> **"HTML"** or **"CSS"**
            - "Python Scripting" -> **"Python"**
            - "OOPS" / "Object Oriented" -> **"OOP"**
            - "Data Structures" -> **"DSA"**
            - "Aptitude" -> **"General Aptitude"**
            - "Introduction" -> **"Behavioral"**

    2.  **DIFFICULTY CLASSIFICATION (STRICT):**
        - **"Easy"**: Definitions, basic syntax, "What is X?", simple explanations.
        - **"Medium"**: Comparisons ("Diff betwen X and Y"), conceptual workings, workflows, standard project questions.
        - **"Hard"**: System design, complex algorithms, optimization strategies, debugging complex scenarios, writing live code for logic.

    3.  **QUESTION TYPE CATEGORIZATION:**
        - **"Theory"**: Conceptual questions, definitions, differences.
        - **"Coding"**: Requests to write code, solve logic puzzles, or implement a function live.
        - **"Behavioral"**: HR questions, soft skills, introduction (Only include if relevant to technical role context).
        - **"Project"**: Questions specifically about the candidate's project architecture or implementation.

    4.  **Content Rules:**
        - **Summarize Answers:** Concisely summarize the candidate's answer (2-3 sentences). 
        - **Coding Challenges:** If the interviewer asks to write code, capture the request as the Question. Leave the Answer as "" (empty string).
        - **Fragmented Questions:** Synthesize interrupted or multi-line questions into a single clear sentence.
        - **Exclusions:** NO small talk ("Can you hear me?", "Hello").

    📄 **Output Format (strict JSON):**
    {{
      "qa_pairs": [
        {{
          "question": "<Formal Rephrased Question>",
          "answer": "<Summarized Answer or empty string>",
          "relevancy_score": <1-5 Integer>,
          "tech_stack": "<Standardized Tech Name based on Question>",
          "question_type": "<Theory | Coding | Behavioral | Project>",
          "difficulty": "<Easy | Medium | Hard>"
        }}
      ]
    }}

    If no valid pairs are found, return: {{"qa_pairs": []}}
    **Return ONLY the JSON object.**

    ---
    🗒️ **Transcript to Analyze:**
    ---
    {full_transcript_text}
    ---
    """
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Use mistral-large-latest for better instruction following on analysis
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    try:
        with st.spinner("🤖 Sending transcript to Mistral for analysis..."):
            response = requests.post(MISTRAL_CHAT_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            json_str = result['choices'][0]['message']['content']
            
            # Clean markdown if present
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                st.error(f"❌ Mistral produced invalid JSON. Parsing failed: {e}")
                st.warning("Below is the raw response for debugging:")
                st.code(json_str, language="json")
                return None

            qa_list = data.get('qa_pairs', [])
            if not qa_list or not isinstance(qa_list, list):
                st.warning("ℹ️ Mistral found no technical Q&A pairs in the transcript.")
                return pd.DataFrame()
            
            df = pd.DataFrame(qa_list)
            df.insert(0, 'No.', range(1, 1 + len(df)))
            
            # Ensure all expected columns exist
            expected_cols = ['question', 'answer', 'relevancy_score', 'tech_stack', 'question_type', 'difficulty']
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = "N/A"
            
            df = df.rename(columns={
                "question": "Question", 
                "answer": "Answer", 
                "relevancy_score": "Relevancy Score",
                "tech_stack": "Tech Stack", 
                "question_type": "Question Type",
                "difficulty": "Difficulty"
            })
            
            return df[['No.', 'Question', 'Answer', 'Relevancy Score', 'Tech Stack', 'Question Type', 'Difficulty']]

    except Exception as e:
        st.error(f"❌ An unexpected error occurred during Mistral processing: {e}")
        return None

# ==============================================================================
#  STREAMLIT UI & MAIN WORKFLOW
# ==============================================================================

st.set_page_config(page_title="NIAT Analyzer v2", page_icon="🎥", layout="wide")

with st.sidebar:
    st.header("⚙️ Processing Options")
    st.session_state.display_video = st.checkbox(
        "Display video player", value=True,
        help="Check this to display the video player in the UI. The video is always downloaded for processing."
    )

# --- DUAL INPUT METHOD ---
input_df = pd.DataFrame()
is_input_valid = False
COLUMN_NAMES = ['UID', 'NIAT ID', 'Name of the student', 'Company Name', 'Video Link', 'Start Time', 'End Time']

input_method = st.radio("Choose input method:", ["Upload CSV", "Paste Text"], horizontal=True, index=1)

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your data as a CSV file (tab-separated)", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            # Assuming the CSV is tab-separated and has no header
            parsed_df = pd.read_csv(uploaded_file, sep='\t', header=None, names=COLUMN_NAMES, skip_blank_lines=True)
            input_df = parsed_df
            is_input_valid = True
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            is_input_valid = False

elif input_method == "Paste Text":
    drive_links_input = st.text_area(
        "🔗 Paste one interview segment per line (UID, NIAT ID, Name, Company, Link, Start, End - separated by Tab):", 
        height=150,
        placeholder="0fac5c2c-...\tN24H01A0186\tGampala Satabish\tIntelXlabs\thttps://drive.google.com/...\t0\t600",
        key="input_links_area"
    )
    if drive_links_input.strip():
        try:
            parsed_df = pd.read_csv(
                StringIO(drive_links_input), sep='\t', header=None, 
                names=COLUMN_NAMES, skip_blank_lines=True
            )
            input_df = parsed_df
            is_input_valid = True
        except Exception as e:
            st.warning(f"⚠️ Could not parse input. Please ensure format is correct. Error: {e}", icon="❗")
            is_input_valid = False

# --- Common logic for both inputs ---
if is_input_valid and not input_df.empty:
    try:
        # Data cleaning and type conversion
        input_df.dropna(subset=['Company Name', 'Video Link', 'Start Time', 'End Time'], inplace=True)
        input_df['Start Time'] = pd.to_numeric(input_df['Start Time'])
        input_df['End Time'] = pd.to_numeric(input_df['End Time'])
        # Fill optional fields with a placeholder if they are missing
        for col in ['UID', 'NIAT ID', 'Name of the student']:
            if col in input_df.columns:
                input_df[col] = input_df[col].fillna('N/A').astype(str)
            else:
                input_df[col] = 'N/A'

        st.subheader("📋 Live Preview of Parsed Data")
        st.dataframe(input_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error processing the provided data. Please check the columns. Details: {e}")
        is_input_valid = False



if st.button("🚀 Start Processing Segments"):
    if not is_input_valid or input_df.empty:
        st.error("Cannot start processing. Please provide valid input data.", icon="❌")
    elif not check_ffmpeg_installed():
        st.error("⚠️ **FFmpeg not found!** Please install FFmpeg and ensure it's in your system's PATH.", icon="🚫")
    else:
        for i, row in input_df.iterrows():
            # --- Extract all 7 fields ---
            uid = row['UID']
            niat_id = row['NIAT ID']
            student_name = row['Name of the student']
            company_name = row['Company Name'].strip()
            link = row['Video Link'].strip()
            start_time = int(row['Start Time'])
            end_time = int(row['End Time'])

            expander_title = f"▶️ Processing '{company_name}' Segment for {student_name} ({format_timestamp(start_time)} to {format_timestamp(end_time)})"
            with st.expander(expander_title, expanded=True):
                st.write(f"**Student:** `{student_name}` (UID: `{uid}`, NIAT ID: `{niat_id}`)")
                st.write(f"**Company:** `{company_name}`")
                st.write(f"**Link:** `{link}`")
                st.write(f"**Timeframe:** `{start_time}`s to `{end_time}`s")

                file_id = get_file_id_from_link(link)
                if not file_id:
                    st.error("Invalid Google Drive link format. Skipping.", icon="❌")
                    continue
                
                segment_id = f"{file_id}_{start_time}_{end_time}"
                run_dir = setup_persistent_directories(segment_id)
                
                video_path = download_drive_video_with_progress(link, file_id)
                if not video_path:
                    st.error("Failed to download video. Cannot continue with this segment.", icon="❌")
                    continue
                
                if st.session_state.display_video:
                    st.video(video_path, start_time=start_time)

                audio_segment_path = extract_audio_segment_with_progress(video_path, file_id, run_dir, start_time, end_time)
                
                if not audio_segment_path or not os.path.exists(audio_segment_path):
                    st.error("Failed to extract audio segment. Cannot continue.", icon="❌")
                    continue
                
                st.caption("Audio Segment (Extracted for Transcription)")
                st.audio(audio_segment_path)
                
                audio_to_process = audio_segment_path
                
                full_transcript_path = os.path.join(run_dir, "full_transcript.txt")
                if not os.path.exists(full_transcript_path):
                    audio_chunks = split_audio_into_chunks(audio_to_process)
                    if not audio_chunks:
                        continue

                    st.write("🗣️ **Transcription Phase**")
                    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    http_session = requests.Session()
                    http_session.mount("https://", adapter)
                    
                    all_segments = []
                    total_offset_seconds = 0.0
                    progress_text = f"Transcribing {len(audio_chunks)} chunk(s)..."
                    transcribe_progress = st.progress(0, text=progress_text)
                    for j, chunk_path in enumerate(audio_chunks):
                        if segments := transcribe_audio_chunk(chunk_path, http_session):
                            chunk_dur = get_media_duration(chunk_path) or 0
                            for seg in segments:
                                seg['start'] += total_offset_seconds
                                seg['end'] += total_offset_seconds
                                all_segments.append(seg)
                            total_offset_seconds += chunk_dur
                        transcribe_progress.progress((j + 1) / len(audio_chunks), text=f"Transcribing chunk {j+1}/{len(audio_chunks)}")
                        
                    transcribe_progress.empty() 

                    if not all_segments:
                        st.error("Transcription failed. No text was generated. Skipping analysis.", icon="❌")
                        continue
                        
                    raw_transcript_text = "\n".join(
                        f"[{format_timestamp(s['start'] + start_time)}] {s['text']}" for s in all_segments
                    )
                    
                    st.write("✨ Cleaning up transcript to remove echo/feedback artifacts...")
                    final_transcript_text = clean_transcript(raw_transcript_text)

                    save_file(final_transcript_text, run_dir, "full_transcript.txt")
                else:
                    st.success("✅ Full transcript already exists.")
                
                analysis_tab, transcript_tab = st.tabs(["📊 Q&A Analysis", "📝 Full Transcript"])
                with analysis_tab:
                    if os.path.exists(full_transcript_path):
                        with st.spinner("🔗 Creating public transcript link via Gist..."):
                            try:
                                with open(full_transcript_path, 'r', encoding='utf-8') as f:
                                    transcript_content = f.read()
                                gist_filename = f"transcript_{company_name.replace(' ', '_')}_{segment_id}.txt"
                                transcript_public_link = create_gist(gist_filename, transcript_content)
                            except Exception as e:
                                transcript_public_link = None
                                st.warning(f"Could not create transcript link: {e}")
                        
                        qa_df = generate_qa_dataframe(full_transcript_path)

                        if qa_df is not None and not qa_df.empty:
                            st.info(f"Found {len(qa_df)} technical Q&A pairs. Saving to sheet.")
                            qa_df.insert(1, 'UID', uid)
                            qa_df.insert(2, 'NIAT ID', niat_id)
                            qa_df.insert(3, 'Name of the student', student_name)
                            qa_df.insert(4, 'Company Name', company_name)
                            qa_df.insert(5, 'Transcript Link', transcript_public_link or "N/A")
                            
                            st.dataframe(qa_df, use_container_width=True, hide_index=True)
                            
                            gspread_client = get_gspread_client()
                            if gspread_client:
                                worksheet = get_or_create_worksheet(gspread_client, GSHEET_NAME, QA_SUBSHEET_NAME)
                                if worksheet:
                                    write_df_to_gsheet(worksheet, qa_df.copy(), link, file_id, company_name, uid, niat_id, student_name)
                            
                            csv_data = qa_df.to_csv(index=False).encode('utf-8')
                            save_file(qa_df.to_csv(index=False), run_dir, "QnA_Table.csv")
                            st.download_button(
                                label="📥 Download Q&A as CSV", data=csv_data, 
                                file_name=f"QnA_Table_{segment_id}.csv", mime="text/csv", 
                                key=f"download_csv_{segment_id}"
                            )
                        else:
                            st.warning("No technical questions found. A summary row with the transcript link will be saved to the sheet.")
                            
                            summary_data = {
                                'No.': ['N/A'], 'Question': ["No technical questions found in transcript"],
                                'Answer': ['N/A'], 'Relevancy Score': ['N/A'],
                                'Tech Stack': ['N/A'], 'Question Type': ['N/A'],
                                'Difficulty': ['N/A'],
                                'Transcript Link': [transcript_public_link or "N/A"]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)

                            gspread_client = get_gspread_client()
                            if gspread_client:
                                worksheet = get_or_create_worksheet(gspread_client, GSHEET_NAME, QA_SUBSHEET_NAME)
                                if worksheet:
                                    write_df_to_gsheet(worksheet, summary_df.copy(), link, file_id, company_name, uid, niat_id, student_name)
                
                with transcript_tab:
                    if os.path.exists(full_transcript_path):
                        with open(full_transcript_path, 'r', encoding='utf-8') as f:
                            st.text_area("Transcript Content", f.read(), height=400, disabled=True, 
                                         key=f"transcript_{segment_id}")
        
        st.markdown("---")
        st.success("🎉 All segments have been processed!")
        st.balloons()