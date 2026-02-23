import os
from dotenv import load_dotenv
from google import genai
from schemas import ModerationState

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def reporter_node(state: ModerationState) -> dict:
    """
    Synthesizes results into a report using Gemini LLM for a structured prose format.
    """
    print("--- Reporter Node ---")
    
    legal = state.legal
    deepfake = state.deepfake
    fact = state.fact
    
    # Calculate final score (simple average for now)
    scores = [
        legal.legal_issue_score if legal else 0.0,
        deepfake.deepfake_ai_score if deepfake else 0.0,
        fact.fake_score if fact else 0.0
    ]
    
    final_score = min(0.5*scores[0]+0.3*scores[1]+0.3*scores[2],1)
    
    # Prepare data for LLM
    analysis_data = {
        "final_score": final_score,
        "legal": {
            "score": legal.legal_issue_score if legal else 0.0,
            "evidence": legal.legal_issue_evidence if legal and legal.legal_issue_evidence else []
        },
        "deepfake": {
            "score": deepfake.deepfake_ai_score if deepfake else 0.0,
            "evidence": deepfake.deepfake_ai_evidence if deepfake and deepfake.deepfake_ai_evidence else []
        },
        "fact_check": {
            "score": fact.fake_score if fact else 0.0,
            "evidence": fact.fake_evidence if fact and fact.fake_evidence else []
        }
    }

    try:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found.")

        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        당신은 AI 악용 허위 사기 광고 모니터링 분석가입니다. 아래의 분석 데이터를 바탕으로 종합 보고서를 작성해주세요.
        
        보고서는 다음 지침을 따라야 합니다:
        1. 줄 바꿈 문자 등 없이 **줄글 형태**로 읽기 좋고 전문적으로 작성할 것.
        2. 법적 문제, 딥페이크/AI 생성 여부, 팩트 체크 결과를 종합적으로 분석할 것.
        3. 단순 수치 나열보다는 각 분석 결과가 의미하는 바를 설명할 것.
        4. 한국어로 작성할 것.
        5. 12줄 정도로 작성할 것.

        분석 데이터:
        {analysis_data}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if response and response.text:
            report = response.text.strip()
        else:
            raise ValueError("Empty response from Gemini")

    except Exception as e:
        print(f"Error calling LLM for report generation: {e}. Falling back to template.")
        # Fallback to simple template
        report = f"""
# Moderation Analysis Report (Fallback)

## Summary
- **Final Risk Score**: {final_score:.2f} / 1.0

## Detailed Analysis
### 1. Legal Issues: {legal.legal_issue_score if legal else 0.0:.2f}
### 2. Deepfake & AI: {deepfake.deepfake_ai_score if deepfake else 0.0:.2f}
### 3. Fact Check: {fact.fake_score if fact else 0.0:.2f}

## Conclusion
{'⚠️ High risk content detected.' if final_score > 0.7 else '✅ Content appears relatively safe.'}
"""

    return {"report": report, "final_score": final_score}
