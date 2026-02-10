from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# Import usage functions
from youtube_utils import download_video, extract_audio, get_transcript
# Import graph
from graph import app as graph_app

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Moderation API", version="1.0")

class AnalyzeRequest(BaseModel):
    youtube_url: str

class AnalyzeResponse(BaseModel):
    legal_issue: bool
    deepfake_issue: bool
    ai_voice_issue: bool

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    url = request.youtube_url
    print(f"Received request for URL: {url}")
    
    try:
        # 1. Download & Process Video
        # Note: These are synchronous operations. In a production server, 
        # these should be offloaded to background tasks or run in a thread pool.
        # For this prototype, we'll run them directly.
        
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
        # invoke is synchronous. 
        result_state = graph_app.invoke(inputs)
        
        # 3. Parse Results
        legal = result_state.get("legal")
        deepfake = result_state.get("deepfake")
        voice = result_state.get("voice")
        report_text = result_state.get("report", "")
        
        # Legal Logic: Any of illegal, fraud, falsehood
        legal_issue = False
        if legal:
            legal_issue = legal.is_illegal or legal.is_fraud or legal.is_falsehood
        
        # 법적으로 통합하여 하나로, deepfake&ai전문가도 하나로, ai_voice는 따로 

        return AnalyzeResponse(
            legal_issue=legal_issue,
            deepfake_issue=deepfake.is_deepfake or deepfake.is_ai_expert,
            ai_voice_issue=voice.is_ai_voice
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
