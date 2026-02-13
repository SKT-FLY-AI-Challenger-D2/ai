import os
import sys
import json
from dotenv import load_dotenv
from google import genai
from schemas import ModerationState, FactResult

# 환경 설정
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def fact_check_node(state: ModerationState) -> dict:
    """
    Analyzes text for factual accuracy using Gemini.
    Returns a temporary/initial FactResult.
    """
    print(f"--- Fact Check Node ---\nProcessing text length: {len(state.input_text)}")
    
    if not state.input_text:
        return {"fact": FactResult(fake_score=0.0, fake_evidence=["No input text provided."])}

    try:
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found. Returning dummy result.")
            return {"fact": FactResult(fake_score=0.0, fake_evidence=["API Key missing."])}

        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        당신은 팩트 체크 전문가입니다. 다음 텍스트의 사실 여부를 판단해주세요.
        출력 형식은 반드시 JSON이어야 합니다
        "fake_score"에는 거짓일 확률, "fake_evidence"에는 거짓이라고 의심되는 근거들을 작성해주세요.
        의심 정도가 낮고 근거가 부족한 경우에는 evidence를 채우지 않아도 괜찮습니다.
        fake evidence 근거는 짧게 나열해주세요. 
        :
        {{
            "fake_score": 0.0 ~ 1.0 (1.0에 가까울수록 거짓 정보일 확률 높음),
            "fake_evidence": ["근거 1", "근거 2", ...] (각 근거는 문자열 리스트 형태)
        }}
        
        텍스트:
        {state.input_text[:5000]}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if not response or not response.text:
            print("Empty response from Gemini")
            return {"fact": FactResult(fake_score=0.0, fake_evidence=["Empty response from Gemini"])}

        text_resp = response.text.strip()
        print(f"DEBUG: Fact Check Response: {text_resp[:100]}...")
        
        # Simple JSON parsing
        json_str = text_resp
        if "```json" in text_resp:
            json_str = text_resp.split("```json")[1].split("```")[0]
        elif "{" in text_resp:
            json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            
        data = json.loads(json_str)
        
        # Ensure fake_evidence is a list
        evidence = data.get("fake_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            evidence = []
        
        result = FactResult(
            fake_score=float(data.get("fake_score", 0.0)),
            fake_evidence=evidence
        )
        
        return {"fact": result}

    except Exception as e:
        print(f"Error in fact_check_node: {e}")
        return {"fact": FactResult(fake_score=0.0, fake_evidence=[f"Error: {str(e)}"])}
