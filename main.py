from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import cv2
import random
from typing import Optional, List
from dotenv import load_dotenv

# Import usage functions
from youtube_utils import download_video, extract_audio, get_transcript
# Import graph
from graph import app as graph_app
from schemas import FactResult, DeepfakeResult, LegalResult
# TASK-06: 준비 상태 확인용 (Chroma·GOOGLE_API_KEY 지연 초기화)
from nodes.legal import check_ready

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Moderation API", version="2.0")


@app.get("/health")
def health_check():
    """프로세스가 요청을 받을 수 있는지만 확인한다. 외부 자원 상태는 보지 않는다."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    """필수 설정(GOOGLE_API_KEY)과 ChromaDB 연결을 실제로 확인한다 (TASK-06).

    실패해도 프로세스는 유지되며, 원인이 해소되면 재시작 없이 다음 호출에서
    다시 성공할 수 있다(legal.py의 지연 초기화가 실패 상태를 캐시하지 않음).
    """
    try:
        check_ready()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ready"}

class AnalyzeRequest(BaseModel):
    youtube_url: str

class AnalyzeResponse(BaseModel):
    is_ad: bool
    legal: Optional[LegalResult]
    deepfake: Optional[DeepfakeResult]
    fact: Optional[FactResult]
    final_score: float
    report: str

def extract_random_frames(video_path: str, num_frames: int = 4) -> List[str]:
    """
    Extracts num_frames random frames from the video and saves them to downloads/frames/.
    """
    print(f"Extracting {num_frames} random frames from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Total frames is 0 or less.")
        cap.release()
        return []

    # Create frames directory
    frames_dir = os.path.join("downloads", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Pick random indices
    frame_indices = random.sample(range(total_frames), min(num_frames, total_frames))
    frame_indices.sort()
    
    extracted_paths = []
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            video_basename = os.path.basename(video_path).split('.')[0]
            frame_name = f"frame_{video_basename}_{i}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
            print(f"Extracted frame {i+1} at index {idx} to {frame_path}")
        else:
            print(f"Failed to extract frame at index {idx}")

    cap.release()
    return extracted_paths

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    url = request.youtube_url
    print(f"Received request for URL: {url}")
    
    try:
        # 1. Download & Process Video
        print("Downloading video...")
        video_path = download_video(url)
        
        print("Extracting audio...")
        audio_path = extract_audio(video_path)
        
        print("Fetching transcript...")
        input_text = get_transcript(url, audio_path=audio_path)
        
        # print("Extracting random frames...")
        # frame_paths = extract_random_frames(video_path, num_frames=4)
        
        inputs = {
            "input_text": input_text,
            "video_path": video_path,
            "audio_path": audio_path,
            # "frame_paths": frame_paths
        }
        
        # 2. Run Graph
        print("Running analysis graph...")
        result_state = graph_app.invoke(inputs)
        
        # 3. Parse Results
        return AnalyzeResponse(
            is_ad=result_state.get("is_ad", False),
            legal=result_state.get("legal"),
            deepfake=result_state.get("deepfake"),
            fact=result_state.get("fact"),
            final_score=result_state.get("final_score", 0.0),
            report=result_state.get("report", "")
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
