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

# [수정 포인트] Docker ChromaDB 서버 설정
CHROMA_HOST = "localhost"
CHROMA_PORT = 8002
COLLECTION_NAME = "legal_documents"

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

# 검색 결과 개수를 k=8 정도로 유지하여 충분한 법적 근거 확보
retriever = vector_db.as_retriever(search_kwargs={"k": 8})

# 3. LLM 설정 (Gemini 2.0 Flash)
# 모델명은 사용 가능한 최신 버전으로 확인 필요
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # gemini-2.5-flash가 아직 출시 전일 수 있으므로 2.0 권장
    temperature=0, 
    google_api_key=api_key
)

# 4. 통합 프롬프트 정의 (동일 유지)
analysis_template = """
당신은 유튜브 콘텐츠의 법적 리스크를 1차적으로 선별하는 **'법률 위반 심사관'**입니다.
제공된 **[법률/규정 근거]**를 바탕으로 **[유튜브 스크립트]**를 분석하세요.

[법률/규정 근거]:
{context}

[유튜브 스크립트]:
{script}

**⚖️ 심사 및 판정 가이드라인 (Prudent Review):**
(이하 가이드라인 및 JSON 형식 지침 동일)
...
"""

analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
final_chain = analysis_prompt | llm | StrOutputParser()

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
        search_query = script[:1500] 
        # Docker 서버를 통해 검색 수행
        retrieved_docs = retriever.invoke(search_query)
        
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
    try:
        full_report = final_chain.invoke({
            "context": context_text[:8000], 
            "script": script[:12000] 
        })

        json_str_match = re.search(r'```json\s*(.*?)\s*```', full_report, re.DOTALL)
        
        if json_str_match:
            try:
                json_data = json.loads(json_str_match.group(1).strip())
                evidence = json_data.get("legal_issue_evidence", [])
                
                state.legal = LegalResult(
                    legal_issue_score=float(json_data.get("legal_issue_score", 0.0)),
                    legal_issue_evidence=evidence if isinstance(evidence, list) else [str(evidence)]
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

    except Exception as e:
        print(f"❌ 분석 단계 오류 발생: {e}")

    return {"legal": state.legal}