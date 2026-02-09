import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

def download_video(url, output_dir="downloads"):
    """
    Downloads video from YouTube URL to the specified directory.
    Returns the path to the downloaded video file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
        
    return video_path

def extract_audio(video_path, output_dir="downloads"):
    """
    Extracts audio from the video file.
    Returns the path to the extracted audio file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.basename(video_path)
    name, ext = os.path.splitext(filename)
    audio_path = os.path.join(output_dir, f"{name}.mp3")
    
    # Use ffmpeg via os.system or specialized lib. 
    # moviepy is installed, let's use it for simplicity/cross-platform.
    from moviepy import VideoFileClip
    
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio with moviepy: {e}")
        return None

import google.generativeai as genai
import time

def transcribe_with_gemini(audio_path):
    """
    Transcribes audio file using Gemini.
    """
    print(f"Transcribing audio with Gemini: {audio_path}")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
         print("Warning: GOOGLE_API_KEY not found. Skipping transcription.")
         return "No transcript available (Missing API Key)."

    try:
        genai.configure(api_key=api_key)
        
        # Upload file
        audio_file = genai.upload_file(path=audio_path)
        
        # Wait for processing
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = genai.get_file(audio_file.name)

        if audio_file.state.name == "FAILED":
             raise ValueError(f"Audio processing failed: {audio_file.state.name}")
        
        model = genai.GenerativeModel(model_name="gemini-3-flash-preview")
        
        prompt = "Generate a verbatim transcript of this audio file. Do not include timestamps or speaker labels unless necessary for clarity. Just return the text."
        response = model.generate_content([audio_file, prompt])
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Error transcribing with Gemini: {e}")
        return "No transcript available (Error)."

def get_transcript(url, audio_path=None):
    """
    Fetches the transcript for the given YouTube URL.
    Returns the transcript as a single string.
    If no caption is available and audio_path is provided, uses Gemini to transcribe.
    """
    try:
        # Extract video ID
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in url:
            video_id = url.split("shorts/")[1].split("?")[0]
        else:
            return ""
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        return formatter.format_transcript(transcript)
    except Exception as e:
        print(f"Error fetching transcript from YouTube: {e}")
        
        if audio_path and os.path.exists(audio_path):
            print("Attempting fallback transcription with Gemini...")
            return transcribe_with_gemini(audio_path)
            
        return "No transcript available."
