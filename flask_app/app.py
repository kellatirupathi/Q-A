import os
import datetime
import pandas as pd
from moviepy.editor import VideoFileClip
import sys
import json
from generate_interview_feedback import get_generated_intervew_html_path
from google.oauth2 import service_account
from get_scores import get_final_scores
import gspread
from flask_cors import CORS
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from decouple import config
import boto3
import subprocess
import flask
from helpers import video_to_audio_converter, get_segregated_data, get_interivew_feedback, \
    qna_default_prompt, generate_transcript, load_transcript, process_transcript_in_chunks, remove_duplicates, \
    questions_scores_prompt, behaviour_prompt

gemini_api_key = config("GEMINI_API_KEY")

CONFIG = {
    'KEY_FILE_PATH': '../creds.json',
    'SHEET_NAME': 'interview_analysis_from_extension',
    'SCOPES': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
}

credentials = Credentials.from_service_account_file(CONFIG['KEY_FILE_PATH'], scopes=CONFIG['SCOPES'])
gc = gspread.authorize(credentials)


# AWS S3 configuration
s3 = boto3.client('s3', aws_access_key_id=config("AWS_ACCESS_KEY"), aws_secret_access_key=config("AWS_SECRET_KEY"))
S3_BUCKET_NAME = "test-diarizaton"
from flask import Flask, request, jsonify,make_response

app = Flask(__name__)
CORS(app)



def upload_analysis_to_sheets(scores_data, user_id, job_id, url, behaviour_data):
    worksheet = gc.open(CONFIG['SHEET_NAME']).sheet1
    existing_data = worksheet.get_all_records()
    df_existing_data = pd.DataFrame(existing_data)
    df_new_data = scores_data
    df_new_data["user_id"] = user_id
    df_new_data["job_id"] = job_id
    df_new_data["interview_link_id"] = url
    df_new_data["creation_datetime"] = datetime.datetime.now()
    if not df_existing_data.empty:
        existing_rows_index = df_existing_data[
            (df_existing_data['job_id'].isin(df_new_data['job_id'])) &
            (df_existing_data['user_id'].isin(df_new_data['user_id']))
            ].index
        if existing_rows_index.empty:
            set_with_dataframe(worksheet, pd.concat([df_existing_data, df_new_data], ignore_index=True),
                               include_index=False)

    else:
        set_with_dataframe(worksheet, pd.concat([df_existing_data, df_new_data], ignore_index=True),
                           include_index=False)
    ia_worksheet = gc.open(CONFIG['SHEET_NAME']).get_worksheet(1)
    ia_existing_data = ia_worksheet.get_all_records()
    ia_df_existing_data = pd.DataFrame(ia_existing_data)
    ia_df_new_data = behaviour_data
    ia_df_new_data["user_id"] = user_id
    ia_df_new_data["job_id"] = job_id
    ia_df_new_data["interview_link_id"] = url
    ia_df_new_data["creation_datetime"] = datetime.datetime.now()
    if not ia_df_existing_data.empty:
        existing_rows_index = ia_df_existing_data[
            (ia_df_existing_data['job_id'].isin(ia_df_new_data['job_id'])) &
            (ia_df_existing_data['user_id'].isin(ia_df_new_data['user_id']))
            ].index
        if existing_rows_index.empty:
            set_with_dataframe(ia_worksheet, pd.concat([ia_df_existing_data, ia_df_new_data], ignore_index=True),
                               include_index=False)

    else:
        set_with_dataframe(ia_worksheet, pd.concat([ia_df_existing_data, ia_df_new_data], ignore_index=True),
                           include_index=False)


def get_scores(qna_data):
    prompt = questions_scores_prompt()
    # print(questions_scores_prompt())
    scores_df = qna_data.copy()
    for index, row in qna_data.iterrows():
        question = str(row["question_text"])
        answer = str(row["answer_text"])
        print(question, answer)
        segragated_data_score = get_segregated_data(prompt, f"question = {question},answer={answer}")
        try:
            scores_data = json.loads(segragated_data_score)
            scores = scores_data[0]
            scores_df.at[index, "content_relevance_and_completeness"] = scores["content_relevance_and_completeness"]
            scores_df.at[index, "accuracy_and_correctness"] = scores[
                "accuracy_and_correctness"]
            scores_df.at[index, "coherency"] = scores[
                "coherency"]
            scores_df.at[index, "depth_of_understanding_and_insight"] = scores[
                "depth_of_understanding_and_insight"]
            scores_df.at[index, "answer_relevancy_score"] = scores[
                "answer_relevancy_score"]
        except:
            print(sys.exc_info())
    return scores_df


def get_video_duration(video_path):
    clip = VideoFileClip(video_path)
    duration_in_seconds = clip.duration
    duration_in_minutes = str(round(duration_in_seconds / 60,2)) + " Mins"
    clip.close()
    return duration_in_minutes

def convert_webm_to_mp4(input_file, frame_rate=1,retry_count=3):
    try:
        output_file = input_file.replace(".webm",".mp4")
        command = [
            'ffmpeg',
            '-i', input_file,
            '-r', str(frame_rate),
            '-c:v', 'libx264',               
            '-preset', 'superfast',
            '-threads', '0',         
            '-tune', 'fastdecode', 
            output_file
        ]
        subprocess.run(command, check=True)
        if os.path.exists(input_file):
            os.remove(input_file)
        return output_file
    
    except:
        if retry_count > 0:
            print("retrying converting")
            new_retry_count = retry_count -1
            convert_webm_to_mp4(input_file,frame_rate=1,retry_count=new_retry_count)
        else:
            print("unable to convert")
        pass

@app.route('/', methods=['POST', "OPTIONS"])
def analyze_interview():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_preflight_response()
        
    try:
        video_link = request.json['video_link']
        candidate_name = request.json['name']
        candidate_email = request.json['email']
        company_name = request.json['company_name']
        interview_date = datetime.datetime.now()
        video_name = video_link.rsplit("/", 1)[1]
        video_file_name = os.path.basename(video_name)
        s3.download_file(S3_BUCKET_NAME, video_name, video_file_name)
        if "webm" in video_name:
            video_file_name = convert_webm_to_mp4(video_file_name)
        duration = get_video_duration(video_file_name)
        audio_file_name = f"{video_file_name.split('.')[0]}.mp3"
        transcript_file_name = f"{video_file_name.split('.')[0]}.srt"
        qna_file_name = f"{video_file_name.split('.')[0]}-qna.csv"
        scores_file_name = f"{video_file_name.split('.')[0]}-scores.csv"
        interview_analysis_file_name = f"{video_file_name.split('.')[0]}.csv"

        video_to_audio_converter(video_file_name, audio_file_name)
        
        if os.path.exists(video_file_name):
            os.remove(video_file_name)

        if not os.path.exists(transcript_file_name):
            generate_transcript(audio_file_name, transcript_file_name)
            if os.path.exists(audio_file_name):
                os.remove(audio_file_name)
        transcript_txt = load_transcript(transcript_file_name)

        prompt = qna_default_prompt()

        if not os.path.exists(qna_file_name):
            segragated_data_qna_json = process_transcript_in_chunks(transcript_txt, prompt)
            qna_df = pd.DataFrame(segragated_data_qna_json)
            final_qna_df = remove_duplicates(qna_df)
            final_qna_df.to_csv(qna_file_name, index=False)
        final_qna_df = pd.read_csv(qna_file_name)
        if not os.path.exists(scores_file_name):
            scores_df = get_scores(final_qna_df)
            scores_df.to_csv(scores_file_name, index=False)
        scores_data = pd.read_csv(scores_file_name)
        # scores_data.to_csv(scores_file_name, index=False)
        question_wise_scores, concept_wise_scores = get_final_scores(scores_file_name)
        scored_csv_data = load_transcript(scores_file_name)
        prompt = behaviour_prompt()

        if not os.path.exists(interview_analysis_file_name):
            segragated_data_behaviour = get_interivew_feedback(prompt, scored_csv_data)
            behaviour_json_data = json.loads(segragated_data_behaviour)
            behaviour_csv = pd.DataFrame(behaviour_json_data)
            behaviour_csv.to_csv(interview_analysis_file_name, index=False)
        behaviour_csv = pd.read_csv(interview_analysis_file_name)
        behaviour_json_data = json.loads(behaviour_csv.to_json(orient="records"))
        

        upload_analysis_to_sheets(scores_data, candidate_name, company_name, video_link, behaviour_csv)
        first_letter = candidate_name[0]
        print(first_letter)
        html_file = get_generated_intervew_html_path(candidate_name,first_letter, company_name, video_link, interview_date, duration,
                                                     candidate_email, concept_wise_scores, question_wise_scores,
                                                     behaviour_json_data
                                                     )
        
        s3.upload_file(html_file, S3_BUCKET_NAME, f"output/{video_file_name.split('.')[0]}.html",
                       ExtraArgs={'ACL': 'public-read', 'ContentType': 'text/html'})
        s3_output_link = f"https://{S3_BUCKET_NAME}.s3.ap-south-1.amazonaws.com/output/{video_file_name.split('.')[0]}.html"
        response = flask.jsonify({'output_link':s3_output_link})

        response.headers.add('Access-Control-Allow-Origin', '*')

        return _corsify_actual_response(
            jsonify({'output_link': s3_output_link}))


    except Exception as e:
        print(sys.exc_info())
        return jsonify({'error': str(e)}), 500
    
def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
    

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=4000)
