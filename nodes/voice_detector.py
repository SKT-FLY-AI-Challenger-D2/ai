import os
import requests
import json
import time
from urllib.parse import urlparse
from schemas import ModerationState, VoiceResult

def voice_detector_node(state: ModerationState) -> dict:
    """
    Analyzes audio for AI-generated voice using undetectable.ai API.
    """
    print(f"--- Voice Detector Node ---\nProcessing audio: {state.audio_path}")
    
    if not state.audio_path:
        print("No audio path provided.")
        return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

    api_key = os.environ.get("UNDETECTABLE_AI_API_KEY")
    if not api_key:
        print("Warning: UNDETECTABLE_AI_API_KEY not found. Returning dummy result.")
        return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

    try:
        # 1. Get Pre-signed URL
        filename = os.path.basename(state.audio_path).replace(" ", "_")
        presign_url = f"https://ai-audio-detect.undetectable.ai/get-presigned-url?file_name={filename}"
        
        headers = {"apikey": api_key}
        response = requests.get(presign_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to get presigned URL: {response.text}")
            return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}
            
        presign_data = response.json()
        if presign_data.get("status") != "success":
             print(f"API Error (presign): {presign_data}")
             return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

        upload_url = presign_data["presigned_url"]
        file_path_for_api = presign_data["file_path"]

        # 2. Upload Audio
        with open(state.audio_path, 'rb') as f:
            audio_data = f.read()
            
        # Determine content type based on extension
        ext = os.path.splitext(filename)[1].lower().replace('.', '')
        content_type = f"audio/{ext}"
        if ext == 'mp3': content_type = 'audio/mpeg' # adjust common types if needed
        
        upload_headers = {
            "Content-Type": content_type,
            "x-amz-acl": "private"
        }
        
        upload_resp = requests.put(upload_url, headers=upload_headers, data=audio_data)
        
        if upload_resp.status_code != 200:
             print(f"Failed to upload audio: {upload_resp.status_code} {upload_resp.text}")
             return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

        # 3. Submit for Detection
        detect_url = "https://ai-audio-detect.undetectable.ai/detect"
        detect_payload = {
            "key": api_key,
            "url": file_path_for_api,
            "document_type": "Audio",
            "analyzeUpToSeconds": 60
        }
        
        detect_resp = requests.post(detect_url, json=detect_payload)
        detect_data = detect_resp.json()
        
        detection_id = detect_data.get("id")
        if not detection_id:
             print(f"Failed to start detection: {detect_data}")
             return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

        # 4. Poll for Results
        query_url = "https://ai-audio-detect.undetectable.ai/query"
        query_payload = {"id": detection_id}
        
        max_retries = 30
        for _ in range(max_retries):
            query_resp = requests.post(query_url, json=query_payload)
            query_data = query_resp.json()
            
            status = query_data.get("status")
            
            if status == "done":
                result_score = query_data.get("result", 0.0)
                details = query_data.get("result_details", {})
                
                # Assuming result > 0.5 means AI voice, user didn't specify threshold
                is_ai_voice = result_score > 0.5 
                
                return {"voice": VoiceResult(
                    is_ai_voice=is_ai_voice,
                    confidence=result_score,
                    details=details
                )}
            elif status == "failed":
                print(f"Detection failed: {query_data}")
                break
            
            time.sleep(2)
            
        print("Detection timed out.")
        return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}

    except Exception as e:
        print(f"Error in voice_detector_node: {e}")
        return {"voice": VoiceResult(is_ai_voice=False, confidence=0.0)}
