import os
import time
import yt_dlp
from dotenv import load_dotenv
from google import genai
from urllib.parse import urlparse, parse_qs

# 1. .env 파일로부터 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_video_id(url):
    """유튜브 URL에서 비디오 ID 추출"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif '/shorts/' in parsed_url.path:
            return parsed_url.path.split('/shorts/')[1].split('/')[0]
    return None

def download_audio(url, output_dir="downloads"):
    """영상 전체에서 오디오(mp3)만 추출하여 다운로드"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 오디오 추출을 위한 yt-dlp 옵션
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True,
    }

    print(f"[INFO] 유튜브 영상에서 음성 데이터를 추출 중입니다...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join(output_dir, f"{info['id']}.mp3")

def transcribe_with_gemini(audio_path):
    """Gemini 2.5 Flash를 사용하여 전체 음성 STT 수행"""
    if not GOOGLE_API_KEY:
        return "[ERROR] .env 파일에 GOOGLE_API_KEY가 설정되지 않았습니다."

    print(f"[INFO] Gemini 2.5 Flash 모델이 음성을 분석하고 있습니다. 잠시만 기다려 주세요...")
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # 오디오 파일 업로드
        audio_file = client.files.upload(file=audio_path)
        
        # 파일 처리(Processing) 완료 대기
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = client.files.get(name=audio_file.name)
        
        if audio_file.state.name != "ACTIVE":
            raise ValueError(f"파일 상태 이상: {audio_file.state.name}")

        # STT를 위한 프롬프트 구성
        prompt = "이 오디오 파일의 모든 내용을 들리는 대로 정확하게 텍스트 스크립트로 작성해줘. 타임스탬프는 생략하고 줄글 형태로 작성해줘."
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[audio_file, prompt]
        )
        
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Gemini STT 수행 중 오류가 발생했습니다: {e}"

def main():
    target_url = input("스크립트를 생성할 유튜브 링크를 입력하세요: ").strip()
    
    if not get_video_id(target_url):
        print("[ERROR] 유효한 유튜브 URL이 아닙니다.")
        return

    audio_path = None
    try:
        # 1. 음성 추출 (공식 자막 API는 사용하지 않음)
        audio_path = download_audio(target_url)
        
        # 2. Gemini STT 수행
        script = transcribe_with_gemini(audio_path)
        
        print("\n" + "="*60)
        print("📜 추출된 전체 음성 스크립트")
        print("="*60)
        print(script)
        print("="*60)

    except Exception as e:
        print(f"[ERROR] 작업 중 오류 발생: {e}")
        
    finally:
        # 3. 작업 완료 후 임시 오디오 파일 삭제
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"[CLEANUP] 임시 파일({os.path.basename(audio_path)})을 삭제했습니다.")

if __name__ == "__main__":
    main()