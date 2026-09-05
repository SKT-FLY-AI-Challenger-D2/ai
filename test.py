import os
import sys
from dotenv import load_dotenv

load_dotenv()

from graph import app
from youtube_utils import download_video, extract_audio, get_transcript

def main() -> bool:
    """전체 파이프라인 스모크 테스트 (TASK-07). 성공하면 True, 실패하면 False를 반환한다.

    비용 발생 지점: download_video(yt-dlp, 무료), get_transcript(자막 없으면 Gemini
    전사, 유료), app.invoke(Gemini 호출 다수, 유료). 실행 전 이 사실을 인지하고
    선택된 대상 영상 하나에만 사용해야 한다.
    """
    print("AI Moderation System (Gemini 3.0 Powered)")
    print("-----------------------------------------")

    # Interactive input for URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL: ").strip()

    if not url:
        print("[FAIL] No URL provided.")
        return False

    print(f"\nProcessing YouTube URL: {url}...")

    try:
        # 1. Download Video
        print("Downloading video...")
        video_path = download_video(url)
        print(f"Video downloaded to: {video_path}")

        # 2. Extract Audio
        print("Extracting audio...")
        audio_path = extract_audio(video_path)
        print(f"Audio extracted to: {audio_path}")

        # 3. Get Transcript (with fallback)
        print("Fetching transcript...")
        input_text = get_transcript(url, audio_path=audio_path)
        print(f"Transcript length: {len(input_text)} chars")

        inputs = {
            "input_text": input_text,
            "video_path": video_path,
            "audio_path": audio_path
        }

        print("\nRunning analysis...")
        result = app.invoke(inputs)
        report = result.get("report", "")

        print("\n" + "="*30)
        print("FINAL REPORT")
        print("="*30)
        print(report)

        # 판정: 사전조건(다운로드·전사)과 최종 리포트 생성이 모두 성공해야 PASS
        ok = bool(video_path) and os.path.exists(video_path) and bool(input_text) and bool(report)
        print(f"\n[{'PASS' if ok else 'FAIL'}] video={bool(video_path)} transcript={bool(input_text)} report={bool(report)}")
        return ok

    except Exception as e:
        print(f"\n[FAIL] Error processing URL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
