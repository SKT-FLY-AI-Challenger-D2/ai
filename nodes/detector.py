from google import genai
import os
import json
import time
from schemas import ModerationState, DeepfakeResult

def detector_node(state: ModerationState) -> dict:
    """
    Analyzes video for deepfakes using Gemini VLM.
    Returns DeepfakeResult.
    """
    print(f"--- Detector Node ---\nProcessing video: {state.video_path}")
    
    if not state.video_path:
        print("No video path provided.")
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["No video provided."])}

    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
             print("Warning: GOOGLE_API_KEY not found. Returning dummy result.")
             return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["API Key missing."])}

        client = genai.Client(api_key=api_key)
        
        # Upload file
        try:
             video_file = client.files.upload(file=state.video_path)
             print(f"Uploaded video: {video_file.name}")
        except Exception as e:
             print(f"Error uploading file: {e}")
             return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=[f"Upload error: {e}"])}
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name != "ACTIVE":
             raise ValueError(f"Video processing failed: {video_file.state.name}")
        
        prompt = """
        이 영상이 딥페이크인지, 인물은 AI 전문가인지 판단해주세요.
        반드시 JSON 구조로만 반환하세요
        'deepfake_ai_score'에는 딥페이크 혹은 AI로 생성된 전문가 의심 정도, 
        'deepfake_ai_evidence'에는 의심되는 근거들을 나열하세요.
        :
        {
            "deepfake_ai_score": 0.0 ~ 1.0,
            "deepfake_ai_evidence": ["근거 1", "근거 2", ...]
        }
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[video_file, prompt]
        )
        
        if not response or not response.text:
            print("Empty response from Gemini")
            return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["Empty response from Gemini"])}

        text_resp = response.text.strip()
        print(f"DEBUG: Detector Response: {text_resp[:100]}...")
        
        try:
             # Try to find JSON block
            json_str = text_resp
            if "```json" in text_resp:
                json_str = text_resp.split("```json")[1].split("```")[0]
            elif "{" in text_resp:
                json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
            data = json.loads(json_str)
            
            # Ensure evidence is list
            evidence = data.get("deepfake_ai_evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            elif not isinstance(evidence, list):
                evidence = []

            result = DeepfakeResult(
                deepfake_ai_score=float(data.get("deepfake_ai_score", 0.0)),
                deepfake_ai_evidence=evidence
            )

        except Exception as e:
            print(f"Error parsing Gemini response: {e}\nRaw response: {text_resp}")
            result = DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=[f"Parsing error: {e}"])

        return {"deepfake": result}

    except Exception as e:
        print(f"Error in detector_node: {e}")
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=[f"Error: {e}"])}
