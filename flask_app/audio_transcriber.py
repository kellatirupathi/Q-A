import os
import re
import requests
from pydub import AudioSegment
from datetime import datetime, timedelta

def adjust_timestamps(file_path, output_path, time_delta_minutes=10):
    time_format = "%H:%M:%S,%f"
    time_delta = timedelta(minutes=time_delta_minutes)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            timestamps = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', line)
            for timestamp in timestamps:
                time_obj = datetime.strptime(timestamp, time_format)
                new_time = time_obj + time_delta
                new_timestamp = new_time.strftime(time_format)[:-3]
                line = line.replace(timestamp, new_timestamp)
            output_file.write(line)

def format_timestamp(seconds):
    """Converts seconds to SRT format 00:00:00,000"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def mistral_json_to_srt(segments, offset_seconds=0):
    """Converts Mistral JSON response to SRT format string"""
    srt_output = ""
    for i, segment in enumerate(segments):
        start_time = segment.get('start', 0) + offset_seconds
        end_time = segment.get('end', 0) + offset_seconds
        text = segment.get('text', '').strip()
        
        srt_output += f"{i + 1}\n"
        srt_output += f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
        srt_output += f"{text}\n\n"
    return srt_output

class WhisperAudioTranscriber:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.api_endpoint = "https://api.mistral.ai/v1/audio/transcriptions"
        self.total_chunks = 0
        self.chunk_size = 0
        self.audio = None
        self.ext = None
        self.chunks_filename = []

    def load_file(self, file_path: str, chunk_size_in_min: int = 10):
        # Load input audio file
        if '.' in file_path:
            self.ext = file_path.split('.')[-1]
        else:
            raise Exception("File does not have extension type")
        try:
            self.audio = AudioSegment.from_mp3(file_path)
        except Exception as e:
            raise e
        
        # Define the chunk size in milliseconds
        chunk_size = chunk_size_in_min * 60 * 1000
        self.chunk_size = chunk_size
        
        # Calculate the total number of chunks
        total_chunks = len(self.audio) // chunk_size + 1
        self.total_chunks = total_chunks
        print(f"Total chunks created: {total_chunks}")
        return self.audio

    def create_audio_chunks(self, chunk_file_name: str = 'chunk') -> bool:
        if not self.total_chunks or not self.chunk_size or not self.audio or not self.ext:
            raise Exception("Please load_file first to create chunks")
        try:
            # Clean up old chunks list
            self.chunks_filename = []
            
            for i in range(self.total_chunks):
                start_time = i * self.chunk_size
                end_time = (i + 1) * self.chunk_size
                chunk = self.audio[start_time:end_time]

                # Export each chunk with a unique filename
                file_name = f"{chunk_file_name}_{i + 1}.{self.ext}"
                chunk.export(file_name, format=self.ext)
                self.chunks_filename.append(file_name)

        except Exception as e:
            print(f"Error creating chunks: {e}")
            return False
        return True

    def start_transcribing(self, output_filename='output_transcript.srt') -> str:
        if len(self.chunks_filename) == 0:
            raise Exception("Please create_audio_chunks first to start transcribing")
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": "voxtral-mini-latest", 
                "timestamp_granularities": ["segment"],
                "response_format": "verbose_json"
            }
            
            with open(output_filename, 'w', encoding="utf-8") as final_file:
                for i in range(self.total_chunks):
                    chunk_file_name = self.chunks_filename[i]
                    print(f"Transcribing chunk {i+1}/{self.total_chunks}: {chunk_file_name}")
                    
                    with open(chunk_file_name, "rb") as audio_file:
                        files = {"file": (os.path.basename(chunk_file_name), audio_file, "audio/mpeg")}
                        
                        try:
                            response = requests.post(
                                self.api_endpoint, 
                                headers=headers, 
                                files=files, 
                                data=data, 
                                timeout=300
                            )
                            response.raise_for_status()
                            result = response.json()
                            
                            segments = result.get("segments", [])
                            
                            # Calculate time offset based on chunk index
                            # i * chunk_size (ms) / 1000 to get seconds
                            offset_seconds = (i * self.chunk_size) / 1000
                            
                            srt_content = mistral_json_to_srt(segments, offset_seconds)
                            final_file.write(srt_content)
                            
                        except Exception as e:
                            print(f"Error transcribing chunk {i}: {e}")
                            
        except Exception as e:
            print(f"Global Transcription Error: {e}")
            return ""
            
        return output_filename