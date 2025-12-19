# import streamlit as st
# import os
# import sys
# import gdown
# import pandas as pd
# import subprocess
# import gc
# import toml
# import re
# import time
# import math
# import glob
# import json
# import requests
# from dotenv import load_dotenv

# # --- AUTO-CONFIGURE UPLOAD LIMITS ---
# def configure_streamlit_upload_limit():
#     config_dir = ".streamlit"
#     config_path = os.path.join(config_dir, "config.toml")
#     target_limit = 4096  # 4GB
    
#     if not os.path.exists(config_dir):
#         os.makedirs(config_dir)
    
#     current_config = {}
#     if os.path.exists(config_path):
#         try:
#             current_config = toml.load(config_path)
#         except:
#             current_config = {}
            
#     server_conf = current_config.get("server", {})
#     current_limit = server_conf.get("maxUploadSize", 200)
    
#     if current_limit < target_limit:
#         if "server" not in current_config:
#             current_config["server"] = {}
#         current_config["server"]["maxUploadSize"] = target_limit
#         with open(config_path, "w") as f:
#             toml.dump(current_config, f)

# configure_streamlit_upload_limit()

# # --- PATH SETUP ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.dirname(current_dir)
# flask_app_path = os.path.join(root_dir, 'flask_app')
# if flask_app_path not in sys.path:
#     sys.path.append(flask_app_path)

# load_dotenv()

# # --- CONSTANTS ---
# DOWNLOAD_DIR = "DownloadedVideos"
# TRANSCRIPT_DIR = "GeneratedTranscripts"
# QNA_DIR = "Q&A"
# CHUNKS_DIR = "AudioChunks" 

# VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv", "webm", "wmv", "flv", "mpeg", "mpg"]
# AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"]

# for d in [DOWNLOAD_DIR, TRANSCRIPT_DIR, QNA_DIR, CHUNKS_DIR]:
#     if not os.path.exists(d):
#         os.makedirs(d)

# st.set_page_config(page_title="Videos & Audios Analyser ", layout="wide")

# # --- PROMPTS ---

# # STRICT FORMATTING PROMPT
# LABELING_PROMPT = """
# You are a Transcription Formatting Engine.
# **INPUT:** Raw, timestamped text.
# **OUTPUT:** Strictly formatted dialogue.

# **RULES:**
# 1. **NO META TALK:** Do not say "Here is the transcript", "Sure", or "Output:". Start directly with the speaker label.
# 2. **NO MARKDOWN BOLD:** Do NOT use `**Candidate:**`. Use plain text `Candidate:`.
# 3. **LABELS:** Use ONLY these labels:
#    - `Interviewer:`
#    - `Candidate:`
# 4. **REMOVE TIMESTAMPS:** Delete all [00:00:00].
# 5. **SPLIT SPEAKERS:** If the speaker changes in the middle of a paragraph, insert a NEWLINE immediately.

# **EXAMPLE INPUT:**
# [00:01] Hi I am John. [00:02] Okay, tell me about React.

# **EXAMPLE OUTPUT:**
# Candidate: Hi I am John.
# Interviewer: Okay, tell me about React.

# **PREVIOUS CONTEXT:**
# {context}

# **CURRENT RAW TEXT:**
# {chunk}
# """

# CUSTOM_QNA_PROMPT = """
# #### Task: Extract Panel Interview Q&A ####

# **Role:** Technical Interview Scraper
# **Goal:** Extract questions asked by ANY interviewer and the answers provided by the CANDIDATE.

# **Input:** A transcript with speaker labels.

# **Instructions:**
# 1. Identify **Technical Questions** asked by anyone labeled as Interviewer, Panelist, or Speaker.
# 2. Identify the **Answer** given by the **Candidate**.
# 3. Ignore administrative chatter ("Can you hear me?", "Next slide", "Hello").
# 4. Consolidate multi-turn answers if the candidate speaks for a long time.

# **Output Format:**
# Return a strictly valid JSON list of objects.
# Example:
# [
#   {
#     "question": "What is polymorphism?",
#     "answer": "It is the ability of an object to take many forms."
#   }
# ]
# """

# # --- UTILITY CLASSES ---

# class ProgressTracker:
#     def __init__(self, total_items, task_name):
#         self.total = total_items
#         self.start_time = time.time()
#         self.task_name = task_name
#         self.progress_bar = st.progress(0, text=f"Starting {task_name}...")

#     def update(self, current_index):
#         elapsed = time.time() - self.start_time
#         items_done = current_index + 1
#         avg_time = elapsed / items_done if items_done > 0 else 0
#         remaining_items = self.total - items_done
#         etr_seconds = avg_time * remaining_items
        
#         if etr_seconds < 60:
#             etr_str = f"{int(etr_seconds)}s"
#         else:
#             m, s = divmod(etr_seconds, 60)
#             etr_str = f"{int(m)}m {int(s)}s"
            
#         pct = min(items_done / self.total, 1.0)
#         self.progress_bar.progress(pct, text=f"{self.task_name}... ({int(pct*100)}%) - ETR: {etr_str}")

#     def complete(self):
#         self.progress_bar.progress(1.0, text=f"{self.task_name} Complete! ✅")
#         time.sleep(0.5)
#         self.progress_bar.empty()

# # --- CORE FUNCTIONS ---

# def get_media_duration(file_path):
#     cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
#     try:
#         return float(subprocess.check_output(cmd).decode().strip())
#     except:
#         return 0

# def convert_to_mp3_optimized(input_path, output_audio_path):
#     try:
#         command = [
#             "ffmpeg", "-i", input_path, "-vn", 
#             "-acodec", "libmp3lame", "-q:a", "2", 
#             "-threads", "auto", "-preset", "ultrafast", 
#             output_audio_path, "-y"
#         ]
#         subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         return True
#     except Exception as e:
#         raise Exception(f"FFmpeg conversion failed: {e}")
#     finally:
#         gc.collect()

# def format_timestamp(seconds):
#     td = pd.Timedelta(seconds=seconds)
#     total_seconds = int(td.total_seconds())
#     hours, remainder = divmod(total_seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#     return f"{hours:02}:{minutes:02}:{seconds:02}"

# def mistral_json_to_text(segments, offset):
#     output = ""
#     for i, seg in enumerate(segments):
#         start = seg.get('start', 0) + offset
#         end = seg.get('end', 0) + offset
#         text = seg.get('text', '').strip()
#         output += f"[{format_timestamp(start)}] {text}\n"
#     return output

# def clean_prefixes(text):
#     """Removes 'Interviewer:', 'Candidate:', etc. from text string."""
#     if not isinstance(text, str):
#         return text
#     cleaned = re.sub(r'^(?:Interviewer|Candidate|Speaker|Panelist)(?:\s+\d+)?\s*:\s*', '', text, flags=re.IGNORECASE)
#     return cleaned.strip()

# def rigorous_clean_ai_output(text):
#     """
#     Sanitizes AI output by removing meta-text and forcing newlines on hidden labels.
#     Fixes: "...text **Candidate:** text..." -> "...text\nCandidate: text..."
#     """
#     # 1. Remove common AI chatter at start of lines
#     text = re.sub(r'(?i)^(Here is|Sure|Output|Transcript|This is).*?:\s*\n', '', text, flags=re.MULTILINE)
    
#     # 2. Normalize Labels (Remove ** and force newline)
#     # Looks for "**Candidate:**" or "Candidate:" or "** Interviewer : **" hidden in text
#     pattern = r'(\n)?\s*(\*\*|)?(Candidate|Interviewer|Speaker|Panelist)(\s+\d+)?(\*\*|)?\s*:'
    
#     # Replace found pattern with "\nLabel:"
#     cleaned_text = re.sub(pattern, r'\n\3:', text, flags=re.IGNORECASE)
    
#     return cleaned_text

# def merge_consecutive_speaker_lines(text):
#     """
#     Robust merging. First cleans the text, then merges same-speaker blocks.
#     """
#     # Step 1: Rigorous cleaning of artifacts
#     text = rigorous_clean_ai_output(text)
    
#     lines = text.split('\n')
#     merged_output = []
    
#     current_speaker = None
#     current_buffer = []
    
#     speaker_pattern = re.compile(r'^(Candidate|Interviewer(?:\s+\d+)?|Speaker(?:\s+\d+)?|Panelist):\s*(.*)', re.IGNORECASE)
    
#     for line in lines:
#         line = line.strip()
#         if not line: 
#             continue
            
#         match = speaker_pattern.match(line)
        
#         if match:
#             speaker = match.group(1).title()
#             content = match.group(2).strip()
            
#             if speaker == current_speaker:
#                 current_buffer.append(content)
#             else:
#                 if current_speaker and current_buffer:
#                     merged_output.append(f"{current_speaker}: {' '.join(current_buffer)}")
#                 current_speaker = speaker
#                 current_buffer = [content]
#         else:
#             # Continuation of previous speaker
#             if current_speaker:
#                 current_buffer.append(line)
    
#     if current_speaker and current_buffer:
#         merged_output.append(f"{current_speaker}: {' '.join(current_buffer)}")
        
#     return "\n\n".join(merged_output)

# # --- PIPELINE STEPS ---

# def transcribe_with_progress(audio_path, output_srt_path):
#     chunk_len = 600
#     duration = get_media_duration(audio_path)
#     total_chunks = math.ceil(duration / chunk_len) if duration > 0 else 1
    
#     base_name = os.path.basename(audio_path).split('.')[0]
#     chunk_files = []
    
#     for f in glob.glob(os.path.join(CHUNKS_DIR, f"{base_name}_part_*.mp3")):
#         try: os.remove(f)
#         except: pass

#     if duration > chunk_len:
#         split_tracker = ProgressTracker(total_chunks, "Splitting Audio")
#         for i in range(total_chunks):
#             start = i * chunk_len
#             chunk_name = os.path.join(CHUNKS_DIR, f"{base_name}_part_{i}.mp3")
#             cmd = ["ffmpeg", "-i", audio_path, "-ss", str(start), "-t", str(chunk_len), "-acodec", "copy", chunk_name, "-y"]
#             subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             chunk_files.append(chunk_name)
#             split_tracker.update(i)
#         split_tracker.complete()
#     else:
#         chunk_files = [audio_path]

#     API_KEY = os.getenv("MISTRAL_TRANSCRIBE_API_KEY") or os.getenv("MISTRAL_API_KEY")
#     if not API_KEY:
#         st.error("Missing MISTRAL_API_KEY in .env file")
#         return ""

#     headers = {"Authorization": f"Bearer {API_KEY}"}
#     data = {"model": "voxtral-mini-latest", "timestamp_granularities": ["segment"], "response_format": "verbose_json"}
#     api_url = "https://api.mistral.ai/v1/audio/transcriptions"
    
#     full_transcript = ""
#     transcribe_tracker = ProgressTracker(len(chunk_files), "Transcribing Audio")
    
#     with open(output_srt_path, "w", encoding="utf-8") as out_f:
#         for i, chunk_file in enumerate(chunk_files):
#             offset = i * chunk_len
#             try:
#                 with open(chunk_file, "rb") as af:
#                     files = {"file": (os.path.basename(chunk_file), af, "audio/mpeg")}
#                     resp = requests.post(api_url, headers=headers, files=files, data=data, timeout=300)
#                     if resp.status_code == 200:
#                         segments = resp.json().get("segments", [])
#                         text_block = mistral_json_to_text(segments, offset)
#                         out_f.write(text_block)
#                         full_transcript += text_block
#             except Exception:
#                 pass
#             transcribe_tracker.update(i)
#             if chunk_file != audio_path and os.path.exists(chunk_file):
#                 try: os.remove(chunk_file)
#                 except: pass

#     transcribe_tracker.complete()
#     return full_transcript

# def label_auto_detect_with_progress(raw_text, output_path):
#     # Small chunk size for high precision
#     chunk_size = 2500 
#     chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
    
#     API_KEY = os.getenv("MISTRAL_API_KEY")
#     CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
    
#     headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
#     tracker = ProgressTracker(len(chunks), "Intelligent Speaker Labeling (Precision Mode)")
#     full_labeled_text = ""
#     prev_context = "Start of interview."
    
#     for i, chunk in enumerate(chunks):
#         formatted_prompt = LABELING_PROMPT.format(context=prev_context[-500:], chunk=chunk)

#         payload = {
#             "model": "mistral-large-latest",
#             "messages": [{"role": "user", "content": formatted_prompt}],
#             "temperature": 0.0 # Strict
#         }
        
#         try:
#             time.sleep(1) # Stability delay
#             r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=90)
#             if r.status_code == 200:
#                 content = r.json()['choices'][0]['message']['content']
#                 # Initial cleanup
#                 content = content.replace("```", "").replace("text\n", "").strip()
#                 full_labeled_text += content + "\n"
#                 prev_context = content 
#         except:
#             pass
#         tracker.update(i)
        
#     # --- POST-PROCESSING ---
#     # 1. Rigorous Clean (Fixes **Candidate:** and missed newlines)
#     # 2. Merge paragraphs
#     final_clean_text = merge_consecutive_speaker_lines(full_labeled_text)
        
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(final_clean_text)
    
#     tracker.complete()
#     return final_clean_text

# def extract_qna_with_progress(labeled_text, output_csv_path):
#     chunk_size = 4000
#     chunks = [labeled_text[i:i + chunk_size] for i in range(0, len(labeled_text), chunk_size)]
    
#     API_KEY = os.getenv("MISTRAL_API_KEY")
#     CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

#     headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
#     tracker = ProgressTracker(len(chunks), "Extracting Q&A")
#     all_data = []
#     prev_context = ""
    
#     for i, chunk in enumerate(chunks):
#         payload = {
#             "model": "mistral-large-latest",
#             "messages": [
#                 {"role": "system", "content": CUSTOM_QNA_PROMPT + " Return ONLY JSON."},
#                 {"role": "user", "content": f"Context: {prev_context[-200:]}\n\nChunk: {chunk}"}
#             ],
#             "temperature": 0.1,
#             "response_format": {"type": "json_object"}
#         }
#         try:
#             r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
#             if r.status_code == 200:
#                 data = r.json()['choices'][0]['message']['content']
#                 json_data = json.loads(data)
#                 if isinstance(json_data, list): all_data.extend(json_data)
#                 elif isinstance(json_data, dict) and 'questions' in json_data: all_data.extend(json_data['questions'])
#                 prev_context = str(data)
#         except:
#             pass
#         tracker.update(i)
    
#     tracker.complete()
    
#     if all_data:
#         df = pd.DataFrame(all_data)
#         if 'question' in df.columns: 
#             df.rename(columns={'question': 'Question Text', 'answer': 'Answer Text'}, inplace=True)
#         cols = [c for c in ['Question Text', 'Answer Text'] if c in df.columns]
#         df = df[cols]
#         df.dropna(inplace=True)
#         df['Question Text'] = df['Question Text'].apply(clean_prefixes)
#         df['Answer Text'] = df['Answer Text'].apply(clean_prefixes)
#         df = df[df['Question Text'].str.len() > 3] 
#         df.to_csv(output_csv_path, index=False)
#         return df
#     return pd.DataFrame()

# # --- ORCHESTRATOR ---

# def process_media_pipeline(source_path, file_id, original_filename):
#     base_name = file_id
#     audio_path = os.path.join(DOWNLOAD_DIR, f"{base_name}.mp3")
#     raw_transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_raw.txt")
#     labeled_transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_labeled.txt")
#     qna_csv_path = os.path.join(QNA_DIR, f"{base_name}.csv")

#     with st.expander(f"✅ Results: {original_filename}", expanded=True):
#         st.write(f"**ID:** `{file_id}`")

#         if not os.path.exists(audio_path):
#             with st.spinner("Preparing Audio..."):
#                 convert_to_mp3_optimized(source_path, audio_path)
#         st.audio(audio_path)

#         if not os.path.exists(raw_transcript_path):
#             raw_text = transcribe_with_progress(audio_path, raw_transcript_path)
#         else:
#             with open(raw_transcript_path, 'r', encoding='utf-8') as f:
#                 raw_text = f.read()

#         if not os.path.exists(labeled_transcript_path):
#             labeled_text = label_auto_detect_with_progress(raw_text, labeled_transcript_path)
#         else:
#             with open(labeled_transcript_path, "r", encoding="utf-8") as f:
#                 labeled_text = f.read()
        
#         t1, t2 = st.tabs(["📝 Labeled Transcript", "⏱️ Raw Timestamped Transcript"])
        
#         with t1:
#             st.text_area("Speaker Labeled (Cleaned)", labeled_text, height=400, key=f"lab_{file_id}")
#             st.download_button("Download Labeled Transcript", labeled_text, f"{base_name}_labeled.txt", "text/plain", key=f"dl_lab_{file_id}")
            
#         with t2:
#             st.text_area("Raw with Timestamps", raw_text, height=300, key=f"raw_{file_id}")

#         if not os.path.exists(qna_csv_path):
#             qna_df = extract_qna_with_progress(labeled_text, qna_csv_path)
#         else:
#             qna_df = pd.read_csv(qna_csv_path)

#         if not qna_df.empty:
#             st.dataframe(qna_df, use_container_width=True, hide_index=True)
#             csv = qna_df.to_csv(index=False).encode('utf-8')
#             st.download_button(f"Download {original_filename} CSV", csv, f"{base_name}_qna.csv", "text/csv")
#         else:
#             st.warning("No Q&A found.")
            
#     gc.collect()

# # --- UI ---

# st.markdown("### 🗣️ Interview Recording Uploader")
# st.caption("Automatically detects Candidate vs. Interviewers/Panelists. Supports Large Files.")

# tab1, tab2 = st.tabs(["📂 Batch Upload (Local)", "☁️ Batch Drive (Cloud)"])

# with tab1:
#     uploaded_files = st.file_uploader("Select files", type=VIDEO_EXTENSIONS + AUDIO_EXTENSIONS, accept_multiple_files=True)
#     if st.button("Start Local"):
#         if uploaded_files:
#             total = len(uploaded_files)
#             main_bar = st.progress(0, text="Batch Started")
#             for i, f in enumerate(uploaded_files):
#                 st.subheader(f"Processing {i+1}/{total}: {f.name}")
#                 file_id = f.name.replace(" ", "_").rsplit(".", 1)[0]
#                 path = os.path.join(DOWNLOAD_DIR, f.name)
#                 with open(path, "wb") as out:
#                     while True:
#                         chunk = f.read(10 * 1024 * 1024)
#                         if not chunk: break
#                         out.write(chunk)
#                 process_media_pipeline(path, file_id, f.name)
#                 main_bar.progress((i+1)/total, text=f"Completed {i+1}/{total}")
#             st.success("Batch Complete!")

# with tab2:
#     drive_input = st.text_area("Paste Drive IDs/Links (comma or newline separated)")
#     if st.button("Start Drive"):
#         raw_ids = re.split(r'[,\n\s]+', drive_input)
#         ids = [x for x in raw_ids if len(x) > 10]
#         ids = list(set(ids))
#         if ids:
#             total = len(ids)
#             main_bar = st.progress(0, text="Batch Started")
#             for i, did in enumerate(ids):
#                 st.subheader(f"Processing {i+1}/{total}: Drive ID {did}")
#                 path = os.path.join(DOWNLOAD_DIR, f"{did}_drive_raw")
#                 if not os.path.exists(path):
#                     try:
#                         gdown.download(f"https://drive.google.com/uc?id={did}", path, quiet=False)
#                     except:
#                         st.error(f"Failed to download {did}")
#                         continue
#                 if os.path.exists(path):
#                     process_media_pipeline(path, did, f"DriveID: {did}")
#                 main_bar.progress((i+1)/total, text=f"Completed {i+1}/{total}")
#             st.success("Batch Complete!")





import streamlit as st
import os
import sys
import gdown
import pandas as pd
import subprocess
import gc
import toml
import re
import time
import math
import glob
import json
import requests
from dotenv import load_dotenv

# --- AUTO-CONFIGURE UPLOAD LIMITS ---
def configure_streamlit_upload_limit():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    target_limit = 4096  # 4GB
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    current_config = {}
    if os.path.exists(config_path):
        try:
            current_config = toml.load(config_path)
        except:
            current_config = {}
            
    server_conf = current_config.get("server", {})
    current_limit = server_conf.get("maxUploadSize", 200)
    
    if current_limit < target_limit:
        if "server" not in current_config:
            current_config["server"] = {}
        current_config["server"]["maxUploadSize"] = target_limit
        with open(config_path, "w") as f:
            toml.dump(current_config, f)

configure_streamlit_upload_limit()

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
flask_app_path = os.path.join(root_dir, 'flask_app')
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

load_dotenv()

# --- CONSTANTS ---
DOWNLOAD_DIR = "DownloadedVideos"
TRANSCRIPT_DIR = "GeneratedTranscripts"
QNA_DIR = "Q&A"
CHUNKS_DIR = "AudioChunks" 

VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv", "webm", "wmv", "flv", "mpeg", "mpg"]
AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "flac", "aac", "wma"]

for d in [DOWNLOAD_DIR, TRANSCRIPT_DIR, QNA_DIR, CHUNKS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

st.set_page_config(page_title="Videos & Audios Analyzer", layout="wide")

# --- PROMPTS ---

# STRICT FORMATTING PROMPT
LABELING_PROMPT = """
You are a Transcription Formatting Engine.
**INPUT:** Raw, timestamped text.
**OUTPUT:** Strictly formatted dialogue.

**RULES:**
1. **NO META TALK:** Do not say "Here is the transcript", "Sure", or "Output:". Start directly with the speaker label.
2. **NO MARKDOWN BOLD:** Do NOT use `**Candidate:**`. Use plain text `Candidate:`.
3. **LABELS:** Use ONLY these labels:
   - `Interviewer:`
   - `Candidate:`
4. **REMOVE TIMESTAMPS:** Delete all [00:00:00].
5. **SPLIT SPEAKERS:** If the speaker changes in the middle of a paragraph, insert a NEWLINE immediately.

**EXAMPLE INPUT:**
[00:01] Hi I am John. [00:02] Okay, tell me about React.

**EXAMPLE OUTPUT:**
Candidate: Hi I am John.
Interviewer: Okay, tell me about React.

**PREVIOUS CONTEXT:**
{context}

**CURRENT RAW TEXT:**
{chunk}
"""

# UPDATED QNA PROMPT: Removed Score, Kept Relevancy, Specific Order Logic
CUSTOM_QNA_PROMPT = """
#### Task: Extract Panel Interview Q&A with Technical Assessment ####

**Role:** Expert Technical Interview Auditor
**Goal:** Extract questions and answers, then categorize and validate them strictly.

**Input:** A transcript with speaker labels (Interviewer/Candidate).

**Instructions:**
1. **Identify**: Extract questions asked by the Interviewer and answers given by the Candidate.
2. **Question Type**: Classify strictly into ONLY these categories: 
   - 'Theory'
   - 'Coding'
   - 'Self Introduction'
   - 'Project Explanation'
   - 'Behavioral'
3. **Tech Stack**: Identify SPECIFIC tool/language names (e.g., 'React.js', 'Node.js', 'SQL', 'Python', 'AWS', 'Docker'). 
   - **DO NOT** use concepts like 'Authentication', 'Sessions', 'Frontend', 'Backend'. 
   - If the topic is JWT Authentication, the stack is 'JWT'.
4. **Difficulty**: Assess complexity strictly as: 'Easy', 'Medium', 'Hard'.
5. **Relevancy Score (1-10)**: Validate the answer against the question. 
   - 10: Direct, accurate answer.
   - 5: Vague or partially related.
   - 1: Completely dodged or incorrect logic.

**Output Format:**
Return a strictly valid JSON list of objects.
Example:
[
  {
    "question": "What is the difference between EJS and ReactJS?",
    "answer": "React uses a virtual DOM...",
    "relevancy_score": 9,
    "type": "Theory",
    "tech_stack": "React.js, EJS",
    "difficulty": "Easy"
  }
]
"""

# --- UTILITY CLASSES ---

class ProgressTracker:
    def __init__(self, total_items, task_name):
        self.total = total_items
        self.start_time = time.time()
        self.task_name = task_name
        self.progress_bar = st.progress(0, text=f"Starting {task_name}...")

    def update(self, current_index):
        elapsed = time.time() - self.start_time
        items_done = current_index + 1
        avg_time = elapsed / items_done if items_done > 0 else 0
        remaining_items = self.total - items_done
        etr_seconds = avg_time * remaining_items
        
        if etr_seconds < 60:
            etr_str = f"{int(etr_seconds)}s"
        else:
            m, s = divmod(etr_seconds, 60)
            etr_str = f"{int(m)}m {int(s)}s"
            
        pct = min(items_done / self.total, 1.0)
        self.progress_bar.progress(pct, text=f"{self.task_name}... ({int(pct*100)}%) - ETR: {etr_str}")

    def complete(self):
        self.progress_bar.progress(1.0, text=f"{self.task_name} Complete! ✅")
        time.sleep(0.5)
        self.progress_bar.empty()

# --- CORE FUNCTIONS ---

def get_media_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        return float(subprocess.check_output(cmd).decode().strip())
    except:
        return 0

def convert_to_mp3_optimized(input_path, output_audio_path):
    try:
        command = [
            "ffmpeg", "-i", input_path, "-vn", 
            "-acodec", "libmp3lame", "-q:a", "2", 
            "-threads", "auto", "-preset", "ultrafast", 
            output_audio_path, "-y"
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        raise Exception(f"FFmpeg conversion failed: {e}")
    finally:
        gc.collect()

def format_timestamp(seconds):
    td = pd.Timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def mistral_json_to_text(segments, offset):
    output = ""
    for i, seg in enumerate(segments):
        start = seg.get('start', 0) + offset
        end = seg.get('end', 0) + offset
        text = seg.get('text', '').strip()
        output += f"[{format_timestamp(start)}] {text}\n"
    return output

def clean_prefixes(text):
    """Removes 'Interviewer:', 'Candidate:', etc. from text string."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r'^(?:Interviewer|Candidate|Speaker|Panelist)(?:\s+\d+)?\s*:\s*', '', text, flags=re.IGNORECASE)
    return cleaned.strip()

def rigorous_clean_ai_output(text):
    """
    Sanitizes AI output by removing meta-text and forcing newlines on hidden labels.
    """
    # 1. Remove common AI chatter at start of lines
    text = re.sub(r'(?i)^(Here is|Sure|Output|Transcript|This is).*?:\s*\n', '', text, flags=re.MULTILINE)
    
    # 2. Normalize Labels (Remove ** and force newline)
    pattern = r'(\n)?\s*(\*\*|)?(Candidate|Interviewer|Speaker|Panelist)(\s+\d+)?(\*\*|)?\s*:'
    cleaned_text = re.sub(pattern, r'\n\3:', text, flags=re.IGNORECASE)
    
    return cleaned_text

def merge_consecutive_speaker_lines(text):
    """
    Robust merging. First cleans the text, then merges same-speaker blocks.
    """
    text = rigorous_clean_ai_output(text)
    
    lines = text.split('\n')
    merged_output = []
    
    current_speaker = None
    current_buffer = []
    
    speaker_pattern = re.compile(r'^(Candidate|Interviewer(?:\s+\d+)?|Speaker(?:\s+\d+)?|Panelist):\s*(.*)', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line: 
            continue
            
        match = speaker_pattern.match(line)
        
        if match:
            speaker = match.group(1).title()
            content = match.group(2).strip()
            
            if speaker == current_speaker:
                current_buffer.append(content)
            else:
                if current_speaker and current_buffer:
                    merged_output.append(f"{current_speaker}: {' '.join(current_buffer)}")
                current_speaker = speaker
                current_buffer = [content]
        else:
            # Continuation of previous speaker
            if current_speaker:
                current_buffer.append(line)
    
    if current_speaker and current_buffer:
        merged_output.append(f"{current_speaker}: {' '.join(current_buffer)}")
        
    return "\n\n".join(merged_output)

# --- PIPELINE STEPS ---

def transcribe_with_progress(audio_path, output_srt_path):
    chunk_len = 600
    duration = get_media_duration(audio_path)
    total_chunks = math.ceil(duration / chunk_len) if duration > 0 else 1
    
    base_name = os.path.basename(audio_path).split('.')[0]
    chunk_files = []
    
    for f in glob.glob(os.path.join(CHUNKS_DIR, f"{base_name}_part_*.mp3")):
        try: os.remove(f)
        except: pass

    if duration > chunk_len:
        split_tracker = ProgressTracker(total_chunks, "Splitting Audio")
        for i in range(total_chunks):
            start = i * chunk_len
            chunk_name = os.path.join(CHUNKS_DIR, f"{base_name}_part_{i}.mp3")
            cmd = ["ffmpeg", "-i", audio_path, "-ss", str(start), "-t", str(chunk_len), "-acodec", "copy", chunk_name, "-y"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            chunk_files.append(chunk_name)
            split_tracker.update(i)
        split_tracker.complete()
    else:
        chunk_files = [audio_path]

    API_KEY = os.getenv("MISTRAL_TRANSCRIBE_API_KEY") or os.getenv("MISTRAL_API_KEY")
    if not API_KEY:
        st.error("Missing MISTRAL_API_KEY in .env file")
        return ""

    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": "voxtral-mini-latest", "timestamp_granularities": ["segment"], "response_format": "verbose_json"}
    api_url = "https://api.mistral.ai/v1/audio/transcriptions"
    
    full_transcript = ""
    transcribe_tracker = ProgressTracker(len(chunk_files), "Transcribing Audio")
    
    with open(output_srt_path, "w", encoding="utf-8") as out_f:
        for i, chunk_file in enumerate(chunk_files):
            offset = i * chunk_len
            try:
                with open(chunk_file, "rb") as af:
                    files = {"file": (os.path.basename(chunk_file), af, "audio/mpeg")}
                    resp = requests.post(api_url, headers=headers, files=files, data=data, timeout=300)
                    if resp.status_code == 200:
                        segments = resp.json().get("segments", [])
                        text_block = mistral_json_to_text(segments, offset)
                        out_f.write(text_block)
                        full_transcript += text_block
            except Exception:
                pass
            transcribe_tracker.update(i)
            if chunk_file != audio_path and os.path.exists(chunk_file):
                try: os.remove(chunk_file)
                except: pass

    transcribe_tracker.complete()
    return full_transcript

def label_auto_detect_with_progress(raw_text, output_path):
    chunk_size = 2500 
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
    
    API_KEY = os.getenv("MISTRAL_API_KEY")
    CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    tracker = ProgressTracker(len(chunks), "Intelligent Speaker Labeling")
    full_labeled_text = ""
    prev_context = "Start of interview."
    
    for i, chunk in enumerate(chunks):
        formatted_prompt = LABELING_PROMPT.format(context=prev_context[-500:], chunk=chunk)

        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": 0.0
        }
        
        try:
            time.sleep(1)
            r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content']
                content = content.replace("```", "").replace("text\n", "").strip()
                full_labeled_text += content + "\n"
                prev_context = content 
        except:
            pass
        tracker.update(i)
        
    final_clean_text = merge_consecutive_speaker_lines(full_labeled_text)
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_clean_text)
    
    tracker.complete()
    return final_clean_text

def extract_qna_with_progress(labeled_text, output_csv_path):
    # Larger chunk size for context
    chunk_size = 4000
    chunks = [labeled_text[i:i + chunk_size] for i in range(0, len(labeled_text), chunk_size)]
    
    API_KEY = os.getenv("MISTRAL_API_KEY")
    CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    tracker = ProgressTracker(len(chunks), "Extracting, Scoring & Validating Q&A")
    all_data = []
    prev_context = ""
    
    for i, chunk in enumerate(chunks):
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": CUSTOM_QNA_PROMPT + " Return ONLY JSON."},
                {"role": "user", "content": f"Context: {prev_context[-200:]}\n\nChunk: {chunk}"}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        try:
            r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()['choices'][0]['message']['content']
                json_data = json.loads(data)
                
                # Normalize JSON output
                extracted_list = []
                if isinstance(json_data, list): 
                    extracted_list = json_data
                elif isinstance(json_data, dict) and 'questions' in json_data: 
                    extracted_list = json_data['questions']
                elif isinstance(json_data, dict):
                    for k, v in json_data.items():
                        if isinstance(v, list):
                            extracted_list = v
                            break
                
                if extracted_list:
                    all_data.extend(extracted_list)
                    
                prev_context = str(data)
        except:
            pass
        tracker.update(i)
    
    tracker.complete()
    
    # --- DATAFRAME CONSTRUCTION ---
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Standardize column names
        rename_map = {
            'question': 'Question Text',
            'answer': 'Answer Text',
            'relevancy_score': 'Relevancy Score',
            'type': 'Question Type',
            'tech_stack': 'Question Techstack',
            'difficulty': 'Difficulty'
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure all columns exist
        required_cols = ['Question Text', 'Answer Text', 'Relevancy Score', 'Question Type', 'Difficulty', 'Question Techstack']
        for col in required_cols:
            if col not in df.columns:
                df[col] = "N/A"
        
        # Select and Reorder to the REQUESTED Format
        df = df[required_cols]
        
        # Cleaning
        df.dropna(subset=['Question Text', 'Answer Text'], inplace=True)
        df['Question Text'] = df['Question Text'].apply(clean_prefixes)
        df['Answer Text'] = df['Answer Text'].apply(clean_prefixes)
        
        # Filter out junk lines
        df = df[df['Question Text'].str.len() > 3] 
        
        # Save
        df.to_csv(output_csv_path, index=False)
        return df
    return pd.DataFrame()

# --- ORCHESTRATOR ---

def process_media_pipeline(source_path, file_id, original_filename):
    base_name = file_id
    audio_path = os.path.join(DOWNLOAD_DIR, f"{base_name}.mp3")
    raw_transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_raw.txt")
    labeled_transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_labeled.txt")
    qna_csv_path = os.path.join(QNA_DIR, f"{base_name}.csv")

    with st.expander(f"✅ Results: {original_filename}", expanded=True):
        st.write(f"**ID:** `{file_id}`")

        # 1. Audio Conversion
        if not os.path.exists(audio_path):
            with st.spinner("Preparing Audio..."):
                convert_to_mp3_optimized(source_path, audio_path)
        st.audio(audio_path)

        # 2. Transcription
        if not os.path.exists(raw_transcript_path):
            raw_text = transcribe_with_progress(audio_path, raw_transcript_path)
        else:
            with open(raw_transcript_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

        # 3. Labeling
        if not os.path.exists(labeled_transcript_path):
            labeled_text = label_auto_detect_with_progress(raw_text, labeled_transcript_path)
        else:
            with open(labeled_transcript_path, "r", encoding="utf-8") as f:
                labeled_text = f.read()
        
        t1, t2 = st.tabs(["📝 Labeled Transcript", "⏱️ Raw Timestamped Transcript"])
        
        with t1:
            st.text_area("Speaker Labeled (Cleaned)", labeled_text, height=400, key=f"lab_{file_id}")
            st.download_button("Download Labeled Transcript", labeled_text, f"{base_name}_labeled.txt", "text/plain", key=f"dl_lab_{file_id}")
            
        with t2:
            st.text_area("Raw with Timestamps", raw_text, height=300, key=f"raw_{file_id}")

        # 4. Q&A Extraction (Updated without General Score, Specific Column Order)
        if not os.path.exists(qna_csv_path):
            qna_df = extract_qna_with_progress(labeled_text, qna_csv_path)
        else:
            qna_df = pd.read_csv(qna_csv_path)

        if not qna_df.empty:
            st.markdown("### 📊 Interview Assessment")
            
            # Requested Column Order:
            # Question text | answer | Relevency score | Question Type | Difficulty | Question Techstack
            display_cols = ['Question Text', 'Answer Text', 'Relevancy Score', 'Question Type', 'Difficulty', 'Question Techstack']
            
            # Reorder if columns exist
            cols_to_show = [c for c in display_cols if c in qna_df.columns]
            
            st.dataframe(qna_df[cols_to_show], use_container_width=True, hide_index=True)
            
            csv = qna_df[cols_to_show].to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {original_filename} Analysis CSV", csv, f"{base_name}_assessment.csv", "text/csv")
        else:
            st.warning("No Q&A found.")
            
    gc.collect()

# --- UI ---

st.markdown("### Videos & Audios Analyzer")
st.caption("Auto-Transcribe > Detect Speakers > Validate Answers > Extract Tech Stack")

tab1, tab2 = st.tabs(["📂 Batch Upload (Local)", "☁️ Batch Drive (Cloud)"])

with tab1:
    uploaded_files = st.file_uploader("Select files", type=VIDEO_EXTENSIONS + AUDIO_EXTENSIONS, accept_multiple_files=True)
    if st.button("Start Local"):
        if uploaded_files:
            total = len(uploaded_files)
            main_bar = st.progress(0, text="Batch Started")
            for i, f in enumerate(uploaded_files):
                st.subheader(f"Processing {i+1}/{total}: {f.name}")
                file_id = f.name.replace(" ", "_").rsplit(".", 1)[0]
                path = os.path.join(DOWNLOAD_DIR, f.name)
                with open(path, "wb") as out:
                    while True:
                        chunk = f.read(10 * 1024 * 1024)
                        if not chunk: break
                        out.write(chunk)
                process_media_pipeline(path, file_id, f.name)
                main_bar.progress((i+1)/total, text=f"Completed {i+1}/{total}")
            st.success("Batch Complete!")

with tab2:
    drive_input = st.text_area("Paste Drive IDs/Links (comma or newline separated)")
    if st.button("Start Drive"):
        raw_ids = re.split(r'[,\n\s]+', drive_input)
        ids = [x for x in raw_ids if len(x) > 10]
        ids = list(set(ids))
        if ids:
            total = len(ids)
            main_bar = st.progress(0, text="Batch Started")
            for i, did in enumerate(ids):
                st.subheader(f"Processing {i+1}/{total}: Drive ID {did}")
                path = os.path.join(DOWNLOAD_DIR, f"{did}_drive_raw")
                if not os.path.exists(path):
                    try:
                        gdown.download(f"https://drive.google.com/uc?id={did}", path, quiet=False)
                    except:
                        st.error(f"Failed to download {did}")
                        continue
                if os.path.exists(path):
                    process_media_pipeline(path, did, f"DriveID: {did}")
                main_bar.progress((i+1)/total, text=f"Completed {i+1}/{total}")
            st.success("Batch Complete!")
