import os
import json
from google import genai
from schemas import ModerationState

def ad_check_node(state: ModerationState) -> dict:
    """
    Analyzes transcript to determine if it is an advertisement using Gemini.
    Returns {"is_ad": bool}.
    """
    print(f"--- Ad Check Node ---\nChecking transcript for advertisement...")
    
    if not state.input_text:
        print("No transcript provided for ad check.")
        return {"is_ad": False}

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Defaulting to is_ad=False.")
        return {"is_ad": False}

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        당신은 광고 판별 전문가입니다. 다음 영상의 스크립트를 보고 이 영상이 광고인지 아닌지 판별해주세요.
        출력 형식은 반드시 JSON이어야 합니다.
        "is_ad" 필드에 광고이면 true, 아니면 false를 넣어주세요.
        "reason" 필드에 그렇게 판별한 간단한 이유를 작성해주세요.

        스크립트:
        {state.input_text[:5000]}

        결과 예시:
        {{
            "is_ad": true,
            "reason": "제품의 장점을 나열하며 구매를 유도하는 전형적인 광고성 멘트가 포함되어 있음"
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if not response or not response.text:
            print("Empty response from Gemini for ad check.")
            return {"is_ad": False}

        text_resp = response.text.strip()
        print(f"DEBUG: Ad Check Response: {text_resp[:100]}...")
        
        # Simple JSON parsing
        json_str = text_resp
        if "```json" in text_resp:
            json_str = text_resp.split("```json")[1].split("```")[0]
        elif "{" in text_resp:
            json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
        data = json.loads(json_str)
        is_ad = data.get("is_ad", False)
        reason = data.get("reason", "No reason provided.")
        
        print(f"Result: is_ad={is_ad}, reason={reason}")
        
        return {"is_ad": is_ad}

    except Exception as e:
        print(f"Error in ad_check_node: {e}")
        return {"is_ad": False}
