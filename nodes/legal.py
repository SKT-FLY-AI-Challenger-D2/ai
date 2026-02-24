import os
import sys
import json
import re
import time
import chromadb  # Docker 클라이언트 연결을 위해 추가
from dotenv import load_dotenv

# 필수 라이브러리 임포트
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.genai.errors import APIError

from config import settings



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

# 1. 임베딩 모델 로드
print("⚙️ 법률 DB 및 임베딩 모델 로드 중...")
embeddings = GoogleGenerativeAIEmbeddings(
     model="models/gemini-embedding-001",
     google_api_key=api_key 
)

# Docker ChromaDB 서버 설정
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))  # 포트는 숫자로 변환 필요
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "legal_documents")

try:
    # 2. Docker 서버 연결을 위한 HttpClient 생성
    http_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    
    # 3. LangChain용 Chroma 객체 연결 (client 파라미터 사용)
    vector_db = Chroma(
        client=http_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    print(f"✅ Docker ChromaDB 연결 성공 (Port: {CHROMA_PORT})")

except Exception as e:
    print(f"❌ 오류: ChromaDB 서버(Port: {CHROMA_PORT})에 연결할 수 없습니다.")
    print(f"   Docker가 실행 중인지 확인하세요. 에러내용: {e}")
    sys.exit(1)


# 3. LLM 설정 (Gemini 2.0 Flash)
# 모델명은 사용 가능한 최신 버전으로 확인 필요
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", # gemini-2.5-flash가 아직 출시 전일 수 있으므로 2.0 권장
#     temperature=0, 
#     google_api_key=api_key
# )

# 4. 통합 프롬프트 정의 (동일 유지)


analysis_template = """
당신은 유튜브 광고 콘텐츠의 법적 리스크를 1차적으로 선별하는 **'법률 위반 심사관'**입니다.

**[전제]**: 입력되는 스크립트는 모두 광고 콘텐츠입니다. 광고 여부 자체를 판단하지 말고, 위법성 판단에만 집중하세요.

[분석 분야]: {domain}

[법률/규정 근거]:
{context}

[유튜브 스크립트]:
{script}

**⚖️ 분석 순서**
1. 스크립트에서 법적으로 문제될 수 있는 주장이나 표현을 파악하세요.
2. 반드시 제공된 [법률/규정 근거] 안에 있는 법률만 사용하세요. 근거에 없는 법률로는 절대 점수를 주지 마세요.
3. [분석 분야]에 따라 아래 순서로 판단하세요.
   공통 법률/규정은 모든 분야에 추가로 적용됩니다.

[식품/의료/화장품]
- 이 표현이 질병 치료/예방/효능을 주장하는가, 아니면 단순 사용법/정보 전달인가?
  → 단, 사용법 설명이라도 타겟 부위나 증상이 의학적 질환(예: 기미, 검버섯, 야간뇨 등)과 직접 연결되면 간접 효능 주장으로 판단하세요.
- 효능 주장이라면 표현 강도가 어느 수준인가?
- 의약품 오인 유발 표현이 있는가? (예: "약 대신", "보약", "환자분들께")
- 의료인을 자처하며 제품을 홍보하는가? → 있으면 높은 점수를 부여하세요.
- 위 항목 외에도 제공된 [법률/규정 근거]에 비춰 위법으로 판단되는 표현이 있으면 추가로 반영하되, 좀 더 엄격하게 판단하세요.

[금융]
- "100% 수익 보장", "원금 보장", "절대 손실 없음" 같은 표현이 있는가?
  → 있으면 표현 강도와 무관하게 무조건 80점 이상
- 리스크를 명시하지 않고 수익만 강조하는가?
- 단순 가격/비용/수치 정보 전달은 위법이 아닙니다. 금융상품 광고에 해당하는지 먼저 확인하세요.
- 위 항목 외에도 제공된 [법률/규정 근거]에 비춰 위법으로 판단되는 표현이 있으면 추가로 반영하되, 좀 더 엄격하게 판단하세요.

4. 위반 가능성을 판단하고 JSON으로 출력하세요.

**⚖️ 표현 강도 판단 기준 (반드시 준수)**

아래의 표현들을 예시로 삼아 점수를 판단하세요.

🟢 낮음 → 0.4점 미만
- 일반적 사실 전달: "비타민 C가 몸에 좋습니다"
- 가능성 표현: "도움이 될 수 있습니다"
- 사용 방법 설명: "얼굴에 펴 바릅니다" (단, 타겟이 의학적 질환이 아닌 경우)
- 주관적 추천: "성분이 순하고 보습이 잘 됩니다"
- 화장품 본연 기능(보습, 미화, 청결) 범위 내 품질 강조: "유효한 성분을 많이 넣었다"
- 비유적/수사적 표현: "종합 탄력 세트 같은 느낌"
- 제품 카테고리명 언급: "재생 크림을 소개합니다"
- 신체 구조 용어를 사용한 운동법/훈련법 설명: "안구 근육을 강화하는 훈련입니다"
- 연구 출처를 모호하게 언급한 경우: "연구 결과도 많이 나옵니다" → 잡되 낮은 점수(20~30점)

🟡 중간 → 0.4~0.7점
- 출처 기반 정보: "연구에 따르면 ~로 알려져 있습니다"
- 증상/질환과 제품을 직접 연결: "야간뇨로 힘드신 분들께 이 제품을 적극 추천"
- 타겟 부위가 의학적 질환인 사용법 설명: "기미와 검버섯이 있는 부위에 두껍게 바르세요"
- 근거 불명확한 효능 비교: "46년 연구 결과 가장 효과적인 방법"
- 간접 인용 효능 주장 (인용 거리가 가까울수록 높게):
  · 가까운 인용: "효과가 있다고 합니다" → 40~50점
  · 먼 인용: "몸 순환이 달라졌다는 표현이 나왔습니다" → 30~40점
- 의료·질병 무관한 제품 효능 과장(최상급 표현): "이 크림만큼 효과적인 게 없어요" → 40~50점

🔴 높음 → 0.8점 이상
[식품/의료/화장품]
- 효능 단정: "비타민 C가 암을 무조건 치료해줍니다"
- 의약품 대안 표방: "약을 먹어야 할까요? 아닙니다. 바세린이 해결책입니다"
- 질병 치료/예방 단정: "혈당을 100%, 무조건 잡아줍니다"
- 의료인을 자처하며 제품 홍보: "제 환자분들께 드리는 제품입니다"

[금융]
- 수익/원금 보장 단정: "무조건 수익이 납니다", "100% 원금 보장"
- 리스크 은폐: "절대 손실 없는 투자 상품"
- 허가 없는 금융상품 권유: "이 코인 지금 사면 10배 됩니다"

[기타]
- 허위 수치: "100% 만족", "전국 1위"

**⚖️ 주의사항 (반드시 준수)**
* 입력 스크립트는 모두 광고로 전제되어 있으므로, 광고 여부를 판단하지 마세요.
* 사용 방법 설명은 효능 주장이 아닙니다. 단, 타겟이 의학적 질환/증상이면 간접 효능 주장으로 판단하세요.
* 섭취/복용 방법 안내(공복에 마시면 흡수가 잘 됩니다 등)는 효능 주장이 아닙니다.
* 신체 구조 용어나 의학 용어를 사용했다고 해서 무조건 위법으로 잡지 마세요. 실제 효능 단정이나 질병 치료 주장이 있어야 합니다.
* 가격, 수량, 성분 함량, 제품 스펙, 비용 정보 등을 설명하는 것 자체는 위법이 아닙니다.
* 자동차 구매 비용, 유지비 등 일반 소비재 비용 정보를 금융상품 광고로 오인하지 마세요.
* 사실적 수치 나열만으로는 기만적 광고로 보지 마세요. 허위이거나 중요 정보를 명백히 은폐한 경우에만 적용하세요.
* 경제적 이해관계는 스크립트 내에 은폐/누락의 명백한 근거가 있을 때만 점수를 주세요.
* 스크립트의 실제 내용과 무관하게 단순 키워드만으로 법률을 적용하지 마세요.
  예) "대추는 한의학에서"라는 표현만으로 의약품 오인 유발로 잡는 것 (X)
      "대추는 한의학에서 암 치료에 쓰입니다"처럼 구체적 효능을 단정할 때 적용 (O)
* 제품 카테고리명이나 제품명 키워드만으로 효능 주장으로 잡지 마세요.
* 비유적/수사적 표현은 효능 주장으로 잡지 마세요.
* 화장품의 보습, 순함, 미화 등 본연의 기능 범위 내 주관적 추천은 실증자료 요구 대상이 아닙니다.

**⚖️ 출력 규칙**
* 위법이라고 단정짓지 말고 반드시 "~위반 가능성 높음" 또는 "~위반 가능성 있음" 형태로 표현하세요.
* `legal_issue_score` 기준 (0 ~ 1점):
  - 0.8점 이상: 명백한 위반 내용
  - 0.6 ~ 0.8점: 보다 강한 위반 의심
  - 0.4 ~ 0.6점: 위반 의심
  - 0.4점 미만: 문제 없음
  - 점수 양극화 주의: 0점(완전 무해) 또는 0.8점 이상(명백한 위법)의 극단적인 점수만 부여하려는 경향을 피하세요.
   - 완화된 표현("도움", "추천")이나 간접 인용이 포함된 경우, 위에서 제시한 [표현 강도 판단 기준]에 따라 0.3~0.7점 사이에서 문맥에 맞는 세밀한 점수를 부여하세요. 무조건 높은 점수(0.8점 이상)를 주거나 무조건 0점을 주는 것을 피하세요.


* `legal_issue_evidence`는 핵심 근거 2개만 주세요.
  - 반드시 "[법률명] '스크립트 표현' + 위반 이유 한 줄 + 위반 가능성 있음/높음" 형식으로 작성하세요.
  - 항목당 2~3줄을 넘지 마세요.
  - 법률 설명, 긴 부연 설명은 포함하지 마세요.
  - ( 호 / 항 / 조 / 목 ) 은 제외하세요.
  
* 출력 형식은 아래의 Json 형식 예시를 참고하세요.
```json
{{
  "legal_issue_score": 0.8,
  "legal_issue_evidence": [
    "[식품표시광고법] '혈당을 무조건 잡아줍니다' 표현이\n일반 식품을 질병 치료 효능이 있다고 표방하여 소비자 오인 유발,\n위반 가능성 높음",
    "[표시광고법] '100% 효과 보장' 표현이\n객관적 근거 없는 과장 광고에 해당,\n위반 가능성 있음"
  ]
}}
```
"""

analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
# final_chain = analysis_prompt | llm | StrOutputParser()

# 분야 분류용
def classify_domain(script: str) -> str:
    prompt = f"""
    다음 스크립트가 어느 분야의 광고인지 아래 중 하나로만 답하세요.
    [식품, 화장품, 의료, 금융, 공통]
    
    스크립트: {script[:500]}
    
    분야:"""
    for model_name in settings.MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name
                temperature=0, 
                google_api_key=api_key
            )
            result = llm.invoke(prompt).content.strip()
            for domain in ["식품", "화장품", "의료", "금융"]:
                if domain in result:
                    return domain
        except APIError as e:
                print(f"{model_name} API 에러(트래픽 등): {e}. 다음 모델 시도.")
                continue
    return "공통"







#===========================================
# langGraph 노드 함수
#===========================================
def legal_node(state: ModerationState) -> ModerationState:
    print("\n--- ⚖️ Legal Analysis Node (Optimized) ---") 
    script = state.input_text

    if not script or len(script) < 20:
        print("⚠️ 스크립트 내용이 부족하여 분석을 스킵합니다.")
        return state

    print(f"🔍 법률 데이터베이스 검색 및 매칭 중...")
    try:

        domain = classify_domain(script)
        print(f"🏷️ 분류된 분야: {domain}")

        all_retriever = vector_db.as_retriever(search_kwargs={"k": 20})
        all_docs = all_retriever.invoke(script[:1500])

        retrieved_docs = [
            doc for doc in all_docs
            if domain in doc.metadata.get("domain", "공통")
            or "공통" in doc.metadata.get("domain", "공통")
        ][:8]
        context_text = ""
        sources = list(set([doc.metadata.get('source', '알수없음') for doc in retrieved_docs]))
        print(f"   📚 참조 문서 리스트: {sources}")
        
        for doc in retrieved_docs:
            source = doc.metadata.get('source', '알수없음')
            context_text += f"\n[법적근거 출처: {source}]\n{doc.page_content}\n"
        
    except Exception as e:
        print(f"   ⚠️ DB 검색 오류: {e}")
        context_text = "표시광고법 및 공정거래법 일반 원칙을 적용하십시오."

    # (이하 분석 리포트 생성 및 JSON 파싱 로직 동일)
    print("📝 법적 타당성 검토 및 보고서 작성 중...")
    for model_name in settings.MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name
                temperature=0, 
                google_api_key=api_key
            )
            final_chain = analysis_prompt | llm | StrOutputParser()
            full_report = final_chain.invoke({
                "context": context_text[:8000], 
                "script": script[:12000],
                "domain": domain
            })

            json_str_match = re.search(r'```json\s*(.*?)\s*```', full_report, re.DOTALL)
            
            if json_str_match:
                try:
                    json_data = json.loads(json_str_match.group(1).strip())
                    evidence = json_data.get("legal_issue_evidence", [])

                    state.legal = LegalResult(
                        legal_issue_score=float(json_data.get("legal_issue_score", 0.0)),
                        legal_issue_evidence=evidence if isinstance(evidence, list) else [str(evidence)],
                    )
                    
                    print("\n⚖️ [AI 최종 판단 리포트]")
                    print(f"  - 법적 리스크 점수: {state.legal.legal_issue_score}")
                    print(f"  - 핵심 근거: {state.legal.legal_issue_evidence}")
                    print("--------------------------\n")
                    
                except Exception as json_e:
                    print(f"⚠️ JSON 데이터 매핑 실패: {json_e}")
            else:
                print("⚠️ 보고서 내에서 JSON 지표를 찾지 못했습니다.")

            state.report = full_report 
        except APIError as e:
            print(f"{model_name} API 에러(트래픽 등): {e}. 다음 모델 시도.")
            continue

        except Exception as e:
            print(f"❌ 분석 단계 오류 발생: {e}")

    return {"legal": state.legal}