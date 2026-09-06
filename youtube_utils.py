import os
os.environ.pop('NODE_CHANNEL_FD', None)  # ← 추가
import shutil
import tempfile
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from moviepy import VideoFileClip
from google import genai
import time

from config import settings


def _youtube_extractor_args() -> dict:
    """YouTube 추출 옵션 (TASK-07, TASK-16).

    - PO 토큰: 명시값(YOUTUBE_PO_TOKEN)이 있으면 그걸 쓰고, 없으면 bgutil 제공자
      서버(BGUTIL_POT_BASE_URL)에서 자동 발급받는다. 둘 다 없으면 옵션 미지정.
    - player_client는 강제하지 않는다(yt-dlp 기본 선택 + EJS 솔버가 처리).
    """
    args: dict = {}
    if settings.YOUTUBE_PO_TOKEN:
        args["youtube"] = {"po_token": [settings.YOUTUBE_PO_TOKEN]}
    if settings.BGUTIL_POT_BASE_URL:
        # bgutil HTTP 제공자 전용 네임스페이스 (구 youtube:getpot_bgutil_baseurl는 deprecated)
        args["youtubepot-bgutilhttp"] = {"base_url": [settings.BGUTIL_POT_BASE_URL]}
    return args


def _cookie_file() -> str | None:
    """마운트된 쿠키 원본을 건드리지 않도록, 매 호출마다 쓰기 가능한 임시 사본을
    만들어 그 경로를 돌려준다. yt-dlp가 세션 갱신 시 쿠키 파일을 되쓰는데,
    실패/챌린지 응답을 반복 저장하면 원본 세션이 열화되기 때문 (TASK-16).
    """
    src = settings.YOUTUBE_COOKIE_PATH
    if not src or not os.path.exists(src):
        return None
    fd, dst = tempfile.mkstemp(prefix="ytck_", suffix=".txt")
    os.close(fd)
    shutil.copyfile(src, dst)
    return dst

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
    cookie_path = _cookie_file()  # 원본 보호용 임시 사본

    def _common_opts() -> dict:
        o = {
            'quiet': True,
            'cookiefile': cookie_path,
            'extractor_args': _youtube_extractor_args(),
        }
        if settings.YOUTUBE_PROXY:
            o['proxy'] = settings.YOUTUBE_PROXY
        return o

    try:
        end_time_str = time.strftime("%H:%M:%S")
        print("[Youtube Util] 길이 추출 시작", end_time_str)
        # 1️⃣ 먼저 길이만 가져오기 (다운로드 X)
        with yt_dlp.YoutubeDL(_common_opts()) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration", 0)
        end_time_str = time.strftime("%H:%M:%S")
        print("[Youtube Util] 영상 다운로드 시작", end_time_str)
        ydl_opts = {
            **_common_opts(),
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'merge_output_format': 'mp4',
        }

        # 3️⃣ 자르기 필요 시 범위 설정 (YT-DLP Native Clipping)
        if duration > clip_duration:
            half = clip_duration / 2
            start_time = max(0, duration / 2 - half)
            end_time = min(duration, duration / 2 + half)
            print(f"[INFO] Clipping middle {clip_duration}s of video ({start_time}s ~ {end_time}s)...")

            ydl_opts['download_ranges'] = lambda info, ctx: [{
                'start_time': start_time,
                'end_time': end_time,
                'title': 'section',
            }]
            ydl_opts['force_keyframes_at_cuts'] = True
        else:
            print(f"[INFO] Video is short ({duration}s). No clipping needed.")

        # 4️⃣ 실제 다운로드
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_video_path = ydl.prepare_filename(info)
        end_time_str = time.strftime("%H:%M:%S")
        print(f"[SUCCESS] Video saved to {final_video_path} : {end_time_str}")

        return final_video_path
    finally:
        if cookie_path and os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
            except OSError:
                pass


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
        # youtube-transcript-api 1.x는 인스턴스 메서드 fetch()를 쓴다 (구 0.x의
        # classmethod get_transcript는 제거됨). requirements.txt 버전 고정(TASK-08)
        # 이후 이 API 불일치가 드러났다 (TASK-16).
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=['ko', 'en'])
        formatter = TextFormatter()
        return formatter.format_transcript(fetched)
    except Exception as e:
        print(f"Error fetching transcript from YouTube: {e}")
        
        if audio_path and os.path.exists(audio_path):
            print("Attempting fallback transcription with Gemini...")
            return transcribe_with_gemini(audio_path)
            
        return "No transcript available."

# 추후에 다운로드 다 받고 지우도록 수정해야함. 
