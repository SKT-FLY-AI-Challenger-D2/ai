import os
import sys
from dotenv import load_dotenv

load_dotenv()

from graph import app
from youtube_utils import download_video, extract_audio, get_transcript

def main():
    print("AI Moderation System (Gemini 3.0 Powered)")
    print("-----------------------------------------")
    
    # Interactive input for URL
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL: ").strip()
        
    if not url:
        print("No URL provided. Exiting.")
        return

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
        print(f"Transcript: {input_text}")
        print(f"Transcript length: {len(input_text)} chars")
        
        inputs = {
            "input_text": input_text,
            "video_path": video_path,
            "audio_path": audio_path
        }
        
        print("\nRunning analysis...")
        result = app.invoke(inputs)
        
        print("\n" + "="*30)
        print("FINAL REPORT")
        print("="*30)
        print(result["report"])
        
    except Exception as e:
        print(f"\nError processing URL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
