import google.generativeai as genai
import os
import json
import time
from schemas import ModerationState, DetectionResult

def detector_node(state: ModerationState) -> dict:
    """
    Analyzes video for deepfakes using Gemini VLM.
    """
    print(f"--- Detector Node ---\nProcessing video: {state.video_path}")
    
    if not state.video_path:
        print("No video path provided.")
        return {"deepfake": DetectionResult(is_deepfake=False, is_ai_expert=False, confidence=0.0)}

    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
             print("Warning: GOOGLE_API_KEY not found. Returning dummy result.")
             return {"deepfake": DetectionResult(is_deepfake=False, is_ai_expert=False, confidence=0.0)}

        genai.configure(api_key=api_key)
        
        # Upload file (naive implementation, assumes local file)
        # Ideally, we should check if file exists and handle upload properly
        try:
             video_file = genai.upload_file(path=state.video_path)
        except Exception as e:
             print(f"Error uploading file: {e}")
             return {"deepfake": DetectionResult(is_deepfake=False, is_ai_expert=False, confidence=0.0)}
        
        # Wait for processing if necessary (for large videos)
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
             raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        model = genai.GenerativeModel(model_name="gemini-3-flash-preview")
        
        prompt = "Is this video a deepfake? Is the person an AI expert? Return JSON: {is_deepfake: bool, is_ai_expert: bool, confidence: float}"
        response = model.generate_content([video_file, prompt])
        
        # Naive parsing (assuming the model follows instructions perfectly)
        # In production, use structured output or robust parsing
        text_resp = response.text.strip()
        
        try:
             # Try to find JSON block
            json_str = text_resp
            if "```json" in text_resp:
                json_str = text_resp.split("```json")[1].split("```")[0]
            elif "{" in text_resp:
                json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
            data = json.loads(json_str)

            result = DetectionResult(
                is_deepfake=data.get("is_deepfake", False),
                is_ai_expert=data.get("is_ai_expert", False),
                confidence=data.get("confidence", 0.0)
            )

        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            result = DetectionResult(is_deepfake=False, is_ai_expert=False, confidence=0.0)

        return {"deepfake": result}

    except Exception as e:
        print(f"Error in detector_node: {e}")
        return {"deepfake": DetectionResult(is_deepfake=False, is_ai_expert=False, confidence=0.0)}
