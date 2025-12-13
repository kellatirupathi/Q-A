import streamlit as st
import os
import sys
import gdown
import pandas as pd
import json
import subprocess
from dotenv import load_dotenv

# --- FIX: Add flask_app to system path to import helpers ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to root, then into flask_app
root_dir = os.path.dirname(current_dir)
flask_app_path = os.path.join(root_dir, 'flask_app')
if flask_app_path not in sys.path:
    sys.path.append(flask_app_path)

# Import helpers (Only specific processing functions needed)
try:
    from flask_app.audio_transcriber import WhisperAudioTranscriber
    from flask_app.helpers import (
        generate_transcript, load_transcript, 
        process_transcript_in_chunks, remove_duplicates
    )
except ImportError:
    # Fallback if running from different context
    from audio_transcriber import WhisperAudioTranscriber
    from helpers import (
        generate_transcript, load_transcript, 
        process_transcript_in_chunks, remove_duplicates
    )

load_dotenv()

# --- CUSTOM PROMPT FOR 2-COLUMN OUTPUT ---
CUSTOM_QNA_PROMPT = """
#### Task: Extract Interview Questions and Answers ####

**Role:** Technical Interview Scraper
**Goal:** Extract the questions asked by the interviewer and the answers provided by the candidate from the transcript.

**Instructions:**
1. Identify sentences that are questions asked by the interviewer.
2. Identify the response given by the candidate for that specific question.
3. Ignore small talk, pleasantries (hello, how are you), and administrative discussions (can you hear me, etc.).
4. Summarize the answer slightly if it is extremely long, but strictly preserve all technical details and keywords.

**Output Format:**
Return a strictly valid JSON list of objects. Each object must have exactly two keys: "question" and "answer".

Example:
[
  {
    "question": "What is the difference between specific and absolute positioning?",
    "answer": "Absolute positioning removes the element from the document flow, while static is the default behavior."
  },
  {
    "question": "Explain closures in JavaScript.",
    "answer": "A closure is the combination of a function bundled together with references to its surrounding state."
  }
]
"""

# --- CONSTANTS ---
DOWNLOAD_DIR = "DownloadedVideos"
TRANSCRIPT_DIR = "GeneratedTranscripts"
QNA_DIR = "Q&A"

# Ensure directories exist
for d in [DOWNLOAD_DIR, TRANSCRIPT_DIR, QNA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

st.set_page_config(page_title="Single Video Q&A Extractor", layout="wide")

# --- AUDIO EXTRACTION (FFMPEG DIRECT) ---
def extract_audio_ffmpeg(video_path, audio_path):
    """
    Extracts audio from video using FFmpeg subprocess.
    """
    try:
        command = [
            "ffmpeg", 
            "-i", video_path, 
            "-vn", 
            "-acodec", "libmp3lame", 
            "-q:a", "2", 
            audio_path, 
            "-y"
        ]
        # Run ffmpeg command
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e}")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install FFmpeg and add it to your PATH.")

# --- PROCESSING PIPELINE ---
def process_video_pipeline(video_path, file_id):
    """
    Runs analysis pipeline: Audio -> Transcript -> Q&A Table (2 Cols)
    """
    base_name = file_id
    audio_path = os.path.join(DOWNLOAD_DIR, f"{base_name}.mp3")
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.srt")
    qna_csv_path = os.path.join(QNA_DIR, f"{base_name}.csv")

    # 1. Audio Extraction
    st.info("Step 1: Extracting Audio...")
    if not os.path.exists(audio_path):
        with st.spinner("Converting video to audio (FFmpeg)..."):
            try:
                extract_audio_ffmpeg(video_path, audio_path)
                st.success("✅ Audio extracted.")
            except Exception as e:
                st.error(f"Failed to extract audio: {e}")
                return
    else:
        st.write("Using existing audio file.")
    
    st.audio(audio_path)

    # 2. Transcription (Mistral API)
    st.info("Step 2: Generating Transcript (Mistral API)...")
    if not os.path.exists(transcript_path):
        with st.spinner("Transcribing audio via Mistral..."):
            try:
                generate_transcript(audio_path, transcript_path)
                if os.path.exists(transcript_path):
                    st.success("✅ Transcript generated.")
                else:
                    st.error("Transcript file was not created. Check API keys.")
                    return
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                return
    else:
        st.write("Using existing transcript.")
    
    transcript_text = load_transcript(transcript_path)
    with st.expander("View Transcript"):
        st.code(transcript_text)

    # 3. Q&A Extraction (Mistral Chat API)
    st.info("Step 3: Extracting Q&A...")
    qna_df = pd.DataFrame()
    
    # We always re-process or check existence. 
    # If you want to force re-run, remove the os.path.exists check.
    if not os.path.exists(qna_csv_path):
        with st.spinner("Analyzing transcript for Q&A (Mistral Chat)..."):
            try:
                # Use our Custom Prompt here
                qna_json = process_transcript_in_chunks(transcript_text, CUSTOM_QNA_PROMPT)
                
                temp_df = pd.DataFrame(qna_json)
                
                if not temp_df.empty:
                    # Rename columns to match desired output
                    # The prompt asks for 'question' and 'answer' keys
                    temp_df.rename(columns={
                        'question': 'Question Text',
                        'answer': 'Answer Text'
                    }, inplace=True)
                    
                    # Ensure only these 2 columns exist (handle extra columns if AI hallucinates)
                    available_cols = [c for c in ['Question Text', 'Answer Text'] if c in temp_df.columns]
                    qna_df = temp_df[available_cols]
                    
                    # Save
                    qna_df.to_csv(qna_csv_path, index=False)
                    st.success("✅ Q&A Extracted.")
                else:
                    st.warning("No Q&A pairs found.")
            except Exception as e:
                st.error(f"Q&A Extraction failed: {e}")
                return
    else:
        qna_df = pd.read_csv(qna_csv_path)
        st.write("Using existing Q&A data.")
    
    st.markdown("### 📝 Interview Questions & Answers")
    st.dataframe(qna_df, use_container_width=True, hide_index=True)


# --- TABS ---
tab1, tab2 = st.tabs(["📤 Upload Local File", "☁️ Download from Drive"])

# TAB 1: LOCAL UPLOAD
with tab1:
    st.subheader("Upload Video File")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    
    if st.button("Process Uploaded Video"):
        if uploaded_file is not None:
            # Use filename as ID (sanitize it)
            file_id = uploaded_file.name.replace(" ", "_").split(".")[0]
            video_path = os.path.join(DOWNLOAD_DIR, f"{file_id}.mp4")
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            process_video_pipeline(video_path, file_id)
        else:
            st.error("Please upload a file first.")

# TAB 2: DRIVE DOWNLOAD
with tab2:
    st.subheader("Process from Drive ID")
    drive_id = st.text_input("Enter Google Drive File ID:")
    
    if st.button("Download & Process"):
        if drive_id:
            video_path = os.path.join(DOWNLOAD_DIR, f"{drive_id}.mp4")
            url = f"https://drive.google.com/uc?id={drive_id}"
            
            # Download if not exists
            if not os.path.exists(video_path):
                with st.spinner("Downloading from Drive..."):
                    gdown.download(url, video_path, quiet=False, fuzzy=True)
            
            if os.path.exists(video_path):
                process_video_pipeline(video_path, drive_id)
            else:
                st.error("Download failed. Check ID.")
        else:
            st.error("Please enter a Drive ID.")