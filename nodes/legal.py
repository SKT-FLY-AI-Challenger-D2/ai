import os
import sys
import json
import re
from dotenv import load_dotenv

# 필수 라이브러리 임포트
from langchain_chroma import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# [수정] 경로 자동 인식: 하위 폴더에서도 루트의 schemas를 찾을 수 있게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
# 프로젝트 내 schemas.py에서 모델 임포트
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
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# 3. LLM 설정 (Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=api_key
)

# 4. 프롬프트 정의
query_gen_template = """
당신은 '불법 광고 및 기망 행위 키워드 추출 전문가'입니다.
[유튜브 스크립트]를 분석하여, 법률 데이터베이스에서 위반 근거를 찾기 위한 **'수사 키워드'** 5개를 추출하세요.
[유튜브 스크립트]: {script}
[출력 형식]: 키워드1, 키워드2, 키워드3, 키워드4, 키워드5
"""

analysis_template = """
당신은 **'범정부 합동 불법 광고 근절단 AI 전문 수사관'**입니다. 
제공된 **[검색된 법률 근거]**를 바탕으로 **[유튜브 스크립트]** 내의 허위·과장·사기성 광고 행위를 적발하여 보고서를 작성하십시오.

[검색된 법률 근거]: {context}
[유튜브 스크립트]: {script}

**3. ⚖️ 법적 판단 지표 (JSON 데이터)**
반드시 아래 형식을 엄격히 지키세요. confidence는 0.0에서 1.0 사이의 '숫자' 하나만 넣어야 합니다.

```json
{{
  "is_falsehood": true/false,
  "is_fraud": true/false,
  "is_illegal": true/false,
  "confidence": 0.00~1.00,
  "reason": "종합적인 판단 근거 요약"
}}
"""
query_gen_chain = ChatPromptTemplate.from_template(query_gen_template) | llm | StrOutputParser() 
analysis_prompt = ChatPromptTemplate.from_template(analysis_template) 
final_chain = analysis_prompt | llm | StrOutputParser()

# ==============================================================================
# 🤖 LangGraph 노드 함수: legal_node
# ==============================================================================
def legal_node(state: ModerationState) -> ModerationState:
    """
    메인 워크플로우에서 호출되는 법률 분석 노드입니다.
    1. 스크립트에서 위법 키워드 추출
    2. 로컬 법률 DB(Chroma)에서 관련 근거 검색
    3. Gemini 2.0을 통한 수사 보고서 및 JSON 지표 생성
    """
    print("\n--- ⚖️ Legal Analysis Node ---")
    script = state.input_text
    
    # 0. 스크립트 유효성 검사
    if not script or len(script) < 20:
        print("⚠️ 분석할 스크립트 내용이 충분하지 않습니다. 분석을 건너뜁니다.")
        return state

    # 1. 수사 키워드 프로파일링 (위법 징후 추출)
    print("🧠 위법 징후 분석 및 검색 키워드 생성 중...")
    try:
        keywords = query_gen_chain.invoke({"script": script[:3000]})
        print(f"   👉 수사 키워드: [{keywords.strip()}]")
    except Exception as e:
        print(f"   ⚠️ 키워드 추출 실패: {e}")
        keywords = script[:100] # 실패 시 스크립트 앞부분으로 대체
    import time
    time.sleep(10)
    # 2. 로컬 법률 데이터베이스 검색 (RAG)
    print(f"🔍 법률 DB 정밀 검색 중...")
    try:
        # 키워드와 스크립트 앞부분을 조합하여 맥락 파악
        retrieved_docs = retriever.invoke(f"{keywords} {script[:500]}")
        
        context_text = ""
        for doc in retrieved_docs:
            source = doc.metadata.get('source', '알수없음')
            context_text += f"\n--- [문서명: {source}] ---\n{doc.page_content}\n"
        print(f"   ✅ {len(retrieved_docs)}건의 법적 근거 확보 완료.")
    except Exception as e:
        print(f"   ⚠️ DB 검색 중 오류 발생: {e}")
        context_text = "법률 DB 검색에 실패했습니다. 일반적인 법리에 근거하여 판단하세요."

    # 3. 최종 수사 보고서 작성 (LLM 실행)
    print("📝 수사 보고서 및 법적 지표 작성 중...")
    try:
        full_report = final_chain.invoke({
            "context": context_text, 
            "script": script
        })

        # 4. 결과 파싱 및 State 업데이트
        # 보고서 내의 JSON 블록을 찾아 LegalResult 객체로 변환합니다.
        json_str_match = re.search(r'```json\n(.*?)\n```', full_report, re.DOTALL)
        
        if json_str_match:
            json_data = json.loads(json_str_match.group(1).strip())
            
            # Pydantic 모델(LegalResult) 형식에 맞춰 저장
            state.legal = LegalResult(
                is_falsehood=json_data.get("is_falsehood", False),
                is_fraud=json_data.get("is_fraud", False),
                is_illegal=json_data.get("is_illegal", False),
                confidence=json_data.get("confidence", 0.0)
            )
            # 전체 텍스트 보고서는 state.report에 저장 (기존 내용이 있다면 덧붙임 가능)
            state.report = full_report 
            print("✅ 법률 분석 결과가 ModerationState에 성공적으로 저장되었습니다.")
        else:
            print("⚠️ 보고서 내에서 JSON 지표를 찾지 못했습니다.")
            state.report = full_report # JSON은 없어도 텍스트 보고서는 저장

    except Exception as e:
        print(f"❌ 최종 분석 단계 오류: {e}")

    return state