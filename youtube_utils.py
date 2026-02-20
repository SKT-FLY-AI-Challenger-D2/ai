import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from moviepy import VideoFileClip
from google import genai
import time

def download_video(url, output_dir="downloads", clip_duration=60):
    """
    If video duration >= clip_duration:
        download middle clip_duration seconds
    Else:
        download full original video
    Returns path to downloaded video
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1️⃣ 먼저 길이만 가져오기 (다운로드 X)
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)

    # 2️⃣ 옵션 설정 (전체 다운로드)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s_full.%(ext)s'), # 일단 _full로 저장
        'noplaylist': True,
        'merge_output_format': 'mp4',
        'quiet': True,
    }

    print(f"[INFO] Downloading full video (original duration={duration}s)")
    
    # 3️⃣ 실제 다운로드
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        full_video_path = ydl.prepare_filename(info)

    final_video_path = os.path.join(output_dir, f"{info['id']}.mp4")

    # 4️⃣ 자르기 로직 (moviepy 사용)
    if duration > clip_duration:
        from moviepy import VideoFileClip
        print(f"[INFO] Clipping middle {clip_duration}s of video...")
        try:
            half = clip_duration / 2
            start_time = max(0, duration / 2 - half)
            end_time = min(duration, duration / 2 + half)
            
            with VideoFileClip(full_video_path) as video:
                clipped_video = video.subclipped(start_time, end_time)
                clipped_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac", logger=None)
            
            # 원본(_full) 파일 삭제
            if os.path.exists(full_video_path):
                os.remove(full_video_path)
            
            print(f"[SUCCESS] Clipped video saved to {final_video_path}")
            print(f"[INFO] Clipped video duration: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"[ERROR] Manual clipping failed: {e}. Using full video instead.")
            os.rename(full_video_path, final_video_path)
    else:
        print(f"[INFO] Video is short enough. No clipping needed.")
        os.rename(full_video_path, final_video_path)

    return final_video_path


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
    try:
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio with moviepy: {e}")
        return None



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
        client = genai.Client(api_key=api_key)
        
        # Upload file
        audio_file = client.files.upload(file=audio_path)
        
        # Wait for processing
        while audio_file.state.name == "PROCESSING":
            time.sleep(1)
            audio_file = client.files.get(name=audio_file.name)

        if audio_file.state.name != "ACTIVE":
             raise ValueError(f"Audio processing failed with state: {audio_file.state.name}")
        
        prompt = "Generate a verbatim transcript of this audio file. Do not include timestamps or speaker labels unless necessary for clarity. Just return the text."
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[audio_file, prompt]
        )
        
        if not response or not response.text:
             return "No transcript generated (Empty Response)."

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
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path[:7] == '/embed/':
                video_id = parsed_url.path[7:]
            elif parsed_url.path[:3] == '/v/':
                video_id = parsed_url.path[3:]
            elif '/shorts/' in parsed_url.path:
                video_id = parsed_url.path.split('/shorts/')[1].split('/')[0]
            else:
                video_id = None
        else:
            video_id = None

        if not video_id:
             print("Could not extract video ID from URL.")
             raise ValueError("Invalid YouTube URL")
            
        print(f"[INFO] Fetching transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        formatter = TextFormatter()
        return formatter.format_transcript(transcript_list)
    except Exception as e:
        print(f"Error fetching transcript from YouTube: {e}")
        
        if audio_path and os.path.exists(audio_path):
            print("Attempting fallback transcription with Gemini...")
            return transcribe_with_gemini(audio_path)
            
        return "No transcript available."

# 추후에 다운로드 다 받고 지우도록 수정해야함. 