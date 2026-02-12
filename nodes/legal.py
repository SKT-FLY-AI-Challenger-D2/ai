import os
import sys
import json
import re
import time
from dotenv import load_dotenv

# 필수 라이브러리 임포트
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 경로 자동 인식 및 커스텀 스키마 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from schemas import ModerationState, LegalResult

# 0. 환경 설정
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ 오류: GOOGLE_API_KEY가 없습니다. .env 파일을 확인하세요.")
    sys.exit(1)

# 1. 임베딩 모델 및 벡터 DB 연결
print("⚙️ 법률 DB 및 임베딩 모델 로드 중...")
embeddings = GoogleGenerativeAIEmbeddings(
     model="models/gemini-embedding-001",
     google_api_key=api_key 
)

DB_PATH = "./chroma_db_local"
if not os.path.exists(DB_PATH):
    print(f"❌ 오류: '{DB_PATH}' 폴더가 없습니다. laws_embedding.py를 먼저 실행하세요.")
    sys.exit(1)

vector_db = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=embeddings
)
# 검색 결과 개수를 k=7~10 정도로 유지하여 충분한 법적 근거 확보
retriever = vector_db.as_retriever(search_kwargs={"k": 8})

# 3. LLM 설정 (Gemini 2.0 Flash)
# 모델명은 사용 가능한 최신 버전(gemini-2.0-flash 등)으로 확인 필요
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=api_key
)

# 4. 통합 프롬프트 정의 (과장 광고 필터링 로직 강화)
# 4. 통합 프롬프트 정의 (수정됨: 신뢰도 및 마케팅 허용 범위 분석 강화)
# 4. 통합 프롬프트 정의 (법적 근거 강제 주입 버전)
analysis_template = """
당신은 유튜브 콘텐츠의 법적 리스크를 1차적으로 선별하는 **'법률 위반 심사관'**입니다.
제공된 **[법률/규정 근거]**를 바탕으로 **[유튜브 스크립트]**를 분석하세요.

[법률/규정 근거]:
{context}

[유튜브 스크립트]:
{script}

**⚖️ 심사 및 판정 가이드라인 (Prudent Review):**

1.  **위반 의심과 확인 필요의 구분 (중요):**
    * **위반 의심:** 스크립트 내용 자체가 법령을 정면으로 위반하는 경우.
        * 예: 일반 식품을 "암 치료제"라고 소개 (식품위생법 위반 명백)
        * 예: "무조건 수익 보장" (표시광고법 위반 명백)
    * **확인 필요:** 절차적 요건이나 증빙 자료가 영상에 드러나지 않은 경우. **이것을 바로 '불법'으로 간주하지 마십시오.**
        * 예: 자율심의필증이 영상에 안 보임 -> **"심의를 받았는지 확인 필요"** (불법 단정 금지)
        * 예: "특허 받은 성분" 언급 -> **"특허청 등록 여부 사실 확인 필요"**

2.  **법령 매핑 (Legal Mapping):**
    * **식품/건강:** 질병 치료, 예방 효과 표방 -> **[식품위생법 제13조]**
    * **화장품/미용:** 의학적 효능 오인 -> **[화장품법 제13조]**
    * **허위/과장:** 근거 없는 최상급 표현("최고", "유일") -> **[표시광고법 제3조]**

**[JSON 출력 형식 준수]**
'legal_issue_score'에는 법적 문제가 있을 확률, 'legal_issue_evidence'에는 법적 문제가 있다는 근거들을 작성해주세요.
* `legal_issue_score`: 법적 리스크 점수 (0.0 ~ 1.0). 위반이 의심될수록 높음.
* `legal_issue_evidence`: ["판단 근거 1", "판단 근거 2", ...] (가능한 한 구체적인 위반 또는 확인 필요 사항을 작성)

```json
{{
  "legal_issue_score": 0.0,
  "legal_issue_evidence": ["(식품표시광고법 제10조) 영상 내에서 자율심의필증 확인 안됨", "(식품위생법 제13조) 일반 식품을 소화제로 오인 소지"]
}}
```
"""

analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
final_chain = analysis_prompt | llm | StrOutputParser()

#===========================================
# langGraph 노드 함수
#===========================================
def legal_node(state: ModerationState) -> ModerationState:
    """ 
    최적화된 법률 분석 노드: 
    1. LLM 키워드 추출 단계를 생략하고 직접 검색하여 속도 개선 
    2. 중립적 프롬프트를 통해 과잉 적발(False Positive) 방지
    3. DB(Chroma)에서 관련 근거 검색
    4. 법적 타당성 검토 및 JSON 리포트 생성
    """
    print("\n--- ⚖️ Legal Analysis Node (Optimized) ---") 
    script = state.input_text

    # 0. 유효성 검사 (앞부분 코드와 연결)
    if not script or len(script) < 20:
        print("⚠️ 스크립트 내용이 부족하여 분석을 스킵합니다.")
        return state

    # 1. RAG 기반 법률 근거 검색 (출력 간소화)
    print(f"🔍 법률 데이터베이스 검색 및 매칭 중...")
    try:
        search_query = script[:1500] 
        retrieved_docs = retriever.invoke(search_query)
        
        context_text = ""
        # 터미널에는 어떤 문서들을 참고했는지만 간단히 출력
        sources = list(set([doc.metadata.get('source', '알수없음') for doc in retrieved_docs]))
        print(f"   📚 참조 문서 리스트: {sources}")
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', '알수없음')
            # LLM에게는 충분한 정보를 제공
            context_text += f"\n[법적근거 출처: {source}]\n{doc.page_content}\n"
        
    except Exception as e:
        print(f"   ⚠️ DB 검색 오류: {e}")
        context_text = "표시광고법 및 공정거래법 일반 원칙을 적용하십시오."

    # 2. 통합 분석 및 보고서 생성 (프롬프트에서 '법조항 명시' 강조)
    # (analysis_template의 지침에 "관련 법조항(제N조)을 반드시 포함하라"는 내용을 보강하세요)
    print("📝 법적 타당성 검토 및 보고서 작성 중...")
    try:
        # 안전하게 텍스트 길이 제한 (에러 방지)
        full_report = final_chain.invoke({
            "context": context_text[:8000], 
            "script": script[:12000] 
        })

        # 3. JSON 파싱 및 출력
        json_str_match = re.search(r'```json\s*(.*?)\s*```', full_report, re.DOTALL)
        
        if json_str_match:
            try:
                json_data = json.loads(json_str_match.group(1).strip())
                
                # Ensure evidence is list
                evidence = json_data.get("legal_issue_evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                elif not isinstance(evidence, list):
                    evidence = []
                
                state.legal = LegalResult(
                    legal_issue_score=float(json_data.get("legal_issue_score", 0.0)),
                    legal_issue_evidence=evidence
                )
                
                # 판단 근거 출력 시 법조항이 포함된 'reason' 출력
                print("\n⚖️ [AI 최종 판단 리포트]")
                print(f"  - 법적 리스크 점수: {state.legal.legal_issue_score}")
                print(f"  - 핵심 근거: {state.legal.legal_issue_evidence}")
                print("--------------------------\n")
                
            except Exception as json_e:
                print(f"⚠️ JSON 데이터 매핑 실패: {json_e}")
                state.legal = LegalResult(
                    legal_issue_score=0.0,
                    legal_issue_evidence=[f"JSON Parsing Error: {json_e}"]
                )
        else:
            print("⚠️ 보고서 내에서 JSON 지표를 찾지 못했습니다.")
            # fallback
            state.legal = LegalResult(
                legal_issue_score=0.0,
                legal_issue_evidence=["JSON output not found"]
            )

        state.report = full_report 

    except Exception as e:
        print(f"❌ 분석 단계 오류 발생: {e}")
        state.legal = LegalResult(
            legal_issue_score=0.0,
            legal_issue_evidence=[f"Analysis Error: {e}"]
        )

    return {"legal": state.legal}