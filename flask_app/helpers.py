import pandas as pd
import sys
import os
import requests
import json
from dotenv import load_dotenv

# --- UPDATED IMPORT LOGIC ---
try:
    # Attempt relative import first (works when imported as a package)
    from .audio_transcriber import WhisperAudioTranscriber
except ImportError:
    try:
        # Attempt import from flask_app package directly
        from flask_app.audio_transcriber import WhisperAudioTranscriber
    except ImportError:
        # Fallback to direct import (if sys.path is modified)
        from audio_transcriber import WhisperAudioTranscriber
# -----------------------------

load_dotenv()

# Mistral Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-large-latest" 

def video_to_audio_converter(mp4, mp3):
    from moviepy.editor import AudioFileClip
    FILETOCONVERT = AudioFileClip(mp4)
    FILETOCONVERT.write_audiofile(mp3)
    FILETOCONVERT.close()

def get_mistral_completion(prompt, user_content):
    """Helper function to call Mistral Chat API"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt + " Return ONLY valid JSON. Do not include markdown formatting like ```json."},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"} 
    }

    try:
        response = requests.post(MISTRAL_CHAT_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"Mistral API Error: {e}")
        return "[]"

def get_segregated_data(prompt, transcript):
    return get_mistral_completion(prompt, transcript)

def get_interivew_feedback(prompt, transcript):
    return get_mistral_completion(prompt, transcript)

def qna_default_prompt():
    with open('qna_prompt_2.txt', 'r') as file:
        prompt = file.read()
    return prompt

def questions_scores_prompt():
    with open('scores_prompt_new2.txt', 'r') as file:
        prompt = file.read()
    return prompt

def behaviour_prompt():
    with open("interview_analysis_prompt.txt", 'r') as file:
        prompt = file.read()
    return prompt

def load_transcript(transcript_file):
    with open(transcript_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def generate_transcript(mp3_file, transcript_file_name):
    try:
        transcriber = WhisperAudioTranscriber(MISTRAL_API_KEY)
        transcriber.load_file(mp3_file)
        transcriber.create_audio_chunks()
        transcriber.start_transcribing(output_filename=transcript_file_name)
    except Exception as e:
        print(f"Transcription Error: {e}")

def remove_duplicates(df):
    df.drop_duplicates(subset='question_text', keep='last', inplace=True)
    return df

def process_transcript_in_chunks(transcript, prompt, chunk_size=3000):
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    final_output = []
    previous_output = ""
    
    for i, chunk in enumerate(chunks):
        current_output = process_chunk(chunk, previous_output, prompt)
        try:
            previous_output = current_output
            chunk_output = json.loads(current_output)
            if isinstance(chunk_output, list):
                final_output.extend(chunk_output)
            elif isinstance(chunk_output, dict):
                 final_output.append(chunk_output)
        except Exception as e:
            print(f"Error parsing chunk {i}: {e}")
            
    return final_output

def process_chunk(chunk, previous_output, prompt):
    final_transcript = f"Previous Context: {previous_output}\n\nCurrent Transcript Chunk: {chunk}"
    output = get_segregated_data(prompt, final_transcript)
    return output

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)