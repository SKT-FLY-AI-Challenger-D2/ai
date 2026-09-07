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
            # 분석(얼굴 포렌식·전사)엔 480p면 충분하고, YouTube 스로틀링 하에서
            # 1080p60은 다운로드가 10분 이상 걸린다 (TASK-16).
            'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'merge_output_format': 'mp4',
        }

        # 3️⃣ 전체 다운로드 (yt-dlp가 인증·PO토큰·n-sig를 모두 처리한다)
        #
        # 예전에는 download_ranges + force_keyframes_at_cuts로 "가운데 60초"만
        # 받았는데, 이 경로는 yt-dlp가 스트림 URL을 ffmpeg에 넘겨 ffmpeg가 직접
        # googlevideo CDN에 range 요청을 한다. 데이터센터 IP에서는 그 직접 요청이
        # 403 Forbidden으로 막혀 `ffmpeg exited with code 8`이 났다 (TASK-16).
        # → yt-dlp로 480p 전체를 받은 뒤(작아서 부담 적음) 로컬 ffmpeg로 자른다.
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            full_video_path = ydl.prepare_filename(info)
        # merge_output_format으로 확장자가 바뀌었을 수 있다
        if not os.path.exists(full_video_path):
            base = os.path.splitext(full_video_path)[0]
            for ext in ('.mp4', '.mkv', '.webm'):
                if os.path.exists(base + ext):
                    full_video_path = base + ext
                    break
        end_time_str = time.strftime("%H:%M:%S")
        print(f"[SUCCESS] Full video saved to {full_video_path} : {end_time_str}")

        # 4️⃣ 필요하면 로컬에서 가운데 clip_duration초만 잘라낸다
        if duration and duration > clip_duration:
            half = clip_duration / 2
            start_time = max(0, duration / 2 - half)
            base = os.path.splitext(full_video_path)[0]
            clip_path = base + '_clip.mp4'
            print(f"[INFO] Trimming middle {clip_duration}s locally ({start_time:.1f}s ~ {start_time + clip_duration:.1f}s)...")
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time), '-i', full_video_path,
                '-t', str(clip_duration),
                '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac',
                '-movflags', '+faststart', clip_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(clip_path):
                print(f"[WARN] Local trim failed (rc={result.returncode}), using full video.\n{result.stderr[-500:]}")
                return full_video_path
            try:
                os.remove(full_video_path)  # 원본(큰 파일) 정리
            except OSError:
                pass
            print(f"[SUCCESS] Clip saved to {clip_path} : {time.strftime('%H:%M:%S')}")
            return clip_path

        print(f"[INFO] Video is short ({duration}s). No clipping needed.")
        return full_video_path
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
