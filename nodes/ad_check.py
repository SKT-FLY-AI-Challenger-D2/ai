import os
import json
from google import genai
from schemas import ModerationState
from google.genai.errors import APIError

from config import settings

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

    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    제공된 스크립트를 분석하여 후속 팩트체크 및 AI 조작 판별 노드로 전달할 '분석 대상(is_ad: true)'인지 판별하세요.

    [분석 대상 (is_ad: true) 판단 기준]
    아래 항목 중 하나라도 포함되어 있다면 분석 대상으로 확정합니다. 단순 정보 전달 목적이라 하더라도 시청자의 신체적/재산적 피해가 발생할 수 있는 민감한 주제이거나 특정 상품을 다룬다면 모두 true로 판별하세요.
    1. 건강 및 신체 정보: 질병 예방/치료, 다이어트, 영양제, 의약품, 병원 시술 등 건강 관련 정보 전달 및 효능 강조.
    2. 재산 및 금융 정보: 주식, 암호화폐, 부동산, 보험, 대출, 투자 기법, 부업/창업 등 금전적 이익이나 금융 관련 정보 전달.
    3. 상품 및 서비스 추천: 특정 제품(전자기기, 화장품 등), 식품, 서비스 등의 장점을 부각하거나 리뷰, 구매, 이용을 유도하는 내용.

    [분석 제외 대상 (is_ad: false) 판단 기준]
    위 기준에 명확히 해당하지 않는 일반적인 콘텐츠는 "반드시" 제외합니다!!!
    - 스포츠 경기 중계 및 리뷰, 코미디/예능 프로그램, 라디오 방송, 게임 플레이, 단순 일상 브이로그, 순수 교양/정치 사회 뉴스 등.

    [응답 형식]
    반드시 아래 JSON 구조로만 반환하세요:
    {{
    "is_ad": true/false,
    "reason": "[대상]에 대해 [정보전달/효능강조/단순예능 등] 성격을 띠고 있어 [정밀 분석 필요 / 분석 불필요]함"
    }}

    스크립트:
    {state.input_text[:5000]}
    """
    for model_name in settings.MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            if not response or not response.text:
                continue # 빈 응답일 경우 다음 모델 시도

            text_resp = response.text.strip()
            print(f"DEBUG: Ad Check Response ({model_name}): {text_resp[:100]}...")
            
            # Simple JSON parsing
            json_str = text_resp
            if "```json" in text_resp:
                json_str = text_resp.split("```json")[1].split("```")[0]
            elif "{" in text_resp:
                json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                
            data = json.loads(json_str)
            is_ad = data.get("is_ad", False)
            reason = data.get("reason", "No reason provided.")
            
            print(f"Result ({model_name}): is_ad={is_ad}, reason={reason}")
            
            # 성공적으로 응답을 받았으므로 결과 반환 후 함수 종료
            return {"is_ad": is_ad}

        except APIError as e:
            # 에러가 발생하면 다음 모델로 넘어감
            print(f"Error in ad_check_node with model [{model_name}]: {e}")
            continue
        
    # 등록된 모든 모델을 시도했으나 실패한 경우
    print("모든 모델에서 분석에 실패했습니다")
    return {"is_ad": False}
