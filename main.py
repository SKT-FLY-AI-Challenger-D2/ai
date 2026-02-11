from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
from dotenv import load_dotenv

# Import usage functions
from youtube_utils import download_video, extract_audio, get_transcript
# Import graph
from graph import app as graph_app
from schemas import FactResult, DeepfakeResult, LegalResult

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Moderation API", version="2.0")

class AnalyzeRequest(BaseModel):
    youtube_url: str

class AnalyzeResponse(BaseModel):
    legal: Optional[LegalResult]
    deepfake: Optional[DeepfakeResult]
    fact: Optional[FactResult]
    final_score: float
    report: str

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
        
        inputs = {
            "input_text": input_text,
            "video_path": video_path,
            "audio_path": audio_path
        }
        
        # 2. Run Graph
        print("Running analysis graph...")
        result_state = graph_app.invoke(inputs)
        
        # 3. Parse Results
        return AnalyzeResponse(
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
