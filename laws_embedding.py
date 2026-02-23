import os
import re
import glob
import sys
import time
import chromadb
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from chromadb.config import Settings
import uuid

# 환경 변수 로드
load_dotenv()

LAW_FOLDER_PATH = "./laws"
# Docker 서버 설정 (8000번 포트)
# 기존 하드코딩된 부분을 아래와 같이 변경
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "legal_documents")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


DOMAIN_MAP = {
    "AI기본법": ["공통"],
    "금융소비자보호법": ["공통"],
    "기만적인 표시·광고 심사지침": ["공통"],
    "부당한 표시·광고행위의 유형 및 기준 지정고시": ["공통"],
    "부정경쟁방지법": ["공통"],
    "식품표시광고법": ["식품"],
    "약사법": ["식품"],
    "유사수신행위의 규제에 관한 법률": ["금융"],
    "의료기기법": ["의료"],
    "의료법": ["의료", "식품"],
    "자본시장과 금융투자업에 관한 법률": ["금융"],
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률": ["공통"],
    "중요한 표시·광고사항 고시": ["공통"],
    "표시광고법": ["공통"],
    "표시ㆍ광고의 공정화에 관한 법률": ["공통"],
    "형법 제347조": ["공통"],
    "화장품 표시·광고 실증에 관한 규정": ["화장품"],
    "화장품법": ["화장품"],
}

def get_domain(source_name: str) -> list:
    for keyword, domain in DOMAIN_MAP.items():
        if keyword in source_name:
            return domain
    return ["공통"]

# [설명: RateLimitedGeminiEmbeddings 버그 수정]
# 기존: 부모 클래스의 메서드를 오버라이딩하면서 인자(**kwargs)를 누락하거나, 
# self.embed_query()를 리스트 내포로 호출해 재귀 에러(Recursion Error)가 발생하는 버그가 있었습니다.
# 수정: **kwargs를 허용하고, super().embed_documents([t]) 형태로 단일 리스트를 던져 안전하게 처리하게 바꿨습니다.
class RateLimitedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        embeddings = []
        for t in texts:
            # Langchain chroma 내부에서 전달되는 메타데이터 등을 허용하기 위해 **kwargs 전달
            embed_result = super().embed_documents([t], **kwargs)
            embeddings.extend(embed_result)
            time.sleep(0.15)  # RPM 제한 준수
        return embeddings

def clean_text(text):
    text = re.sub(r'안민국', '', text)
    text = re.sub(r'본 판결문은 판결서 인터넷열람 사이트에서.*?금지됩니다\.', '', text, flags=re.DOTALL)
    text = re.sub(r'비실명처리일자\s?:\s?\d{4}-\d{2}-\d{2}', '', text)
    text = re.sub(r'-\s?\d+\s?-', '', text)
    text = re.sub(r'법제처\s+\d+\s+국가법령정보센터', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'무단전재 및 수집, 재배포금지', '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s+오후\s+\d+:\d+', '', text)
    return text.strip()

def process_pdf(file_path):
    file_name = os.path.basename(file_path)
    source_name = os.path.splitext(file_name)[0]
    print(f"📖 읽는 중: {source_name}")

    # [복구: 누락된 Domain 할당 로직 복원]
    # 도메인 매핑 (파일 이름에 기반하여 도메인 부여)
   
    document_domain = ",".join(get_domain(source_name))
    
    loader = PyPDFLoader(file_path)
    try:
        pages = loader.load()
    except Exception as e:
        print(f"⚠️ 로드 실패: {e}")
        return []

    full_text = "\n".join(clean_text(p.page_content) for p in pages)
    docs = []

    article_matches = re.findall(r'(제\s?\d+\s?조\s?\([^)]+\))', full_text)
    is_precedent = re.search(r'주\s*문', full_text) and re.search(r'이\s*유', full_text)

    if len(article_matches) > 5:
        print("👉 [법률 모드]")
        parts = re.split(r'(제\s?\d+\s?조\s?\([^)]+\))', full_text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for i in range(1, len(parts), 2):
            header = parts[i]
            content = parts[i + 1]
            chunks = splitter.split_text(content)
            for chunk in chunks:
                docs.append(Document(
                    page_content=f"[{source_name} {header}]\n{chunk}",
                    metadata={"source": source_name, "type": "law", "domain": document_domain}
                ))
    elif is_precedent:
        print("👉 [판례 모드]")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=300,
            separators=["\n\n", "살피건대", "이에 대하여", "."]
        )
        chunks = splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=f"[{source_name} 판결문 {i+1}]\n{chunk}",
                metadata={"source": source_name, "type": "precedent", "domain": document_domain}
            ))
    else:
        print("👉 [일반 문서 모드]")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            docs.append(Document(
                page_content=f"[{source_name} 문서]\n{chunk}",
                metadata={"source": source_name, "type": "general", "domain": document_domain}
            ))
    return docs

if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY 없음")
        sys.exit(1)

    pdf_files = glob.glob(os.path.join(LAW_FOLDER_PATH, "*.pdf"))
    if not pdf_files:
        print("❌ PDF 없음")
        sys.exit(1)

    all_documents = []
    for pdf in pdf_files:
        all_documents.extend(process_pdf(pdf))

    # [설명: Parent Document Retrieval (PDR) 원리 1단계 - 부모 문서 등록]
    # 이전에는 여기서 얻은 all_documents(큰 덩어리)를 그대로 Chroma에 박고 끝냈습니다.
    # 하지만 이제는 '원본'으로서 Redis에 저장할 것이므로, 각각 고유한 UUID를 발급해 둡니다.
    doc_ids = [str(uuid.uuid4()) for _ in all_documents]

    print(f"\n📦 총 {len(all_documents)}개 원본/부모 문서(Parent Documents) 로드 완료")

    print("⚙️ Gemini Embedding 로딩...")
    embeddings = RateLimitedGeminiEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    print(f"🚀 Docker Chroma DB({CHROMA_HOST}:{CHROMA_PORT}) 및 Redis({REDIS_URL}) 통신 설정 중...")
    
    try:
        # ChromaDB 클라이언트 및 빈 컬렉션(VectorStore) 초기화
        http_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        try:
            http_client.delete_collection(COLLECTION_NAME)
            print("🗑️ 기존 컬렉션 삭제 완료")
        except:
            pass
            
        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            client=http_client
        )
        
        # [설명: PDR 원리 2단계 - 검색용 Child 조각(Chunking) 생성]
        # 거대한 원본 문서를 검색하기 좋게 200자 단위로 잘게 쪼갭니다.
        # 이때 쪼개진 자식(Child) 조각들의 메타데이터에 '부모의 UUID'를 기입하여 꼬리표를 달아둡니다.
        id_key = "doc_id"
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        sub_docs = []
        for i, doc in enumerate(all_documents):
            _id = doc_ids[i]  # 부모의 UUID
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id  # 나는 어느 부모에서 파생되었는지 명시!
            sub_docs.extend(_sub_docs)

        # [설명: PDR 원리 3단계 - Redis 저장소(Docstore) 세팅]
        # 큰 덩어리인 '부모 문서 객체'를 온전히 보관할 별도 저장소를 만듭니다.
        import json
        from langchain_classic.storage.encoder_backed import EncoderBackedStore
        
        # Redis는 파이썬 객체를 그대로 못 넣으므로 JSON으로 인코딩/디코딩 해주는 로직을 감쌉니다.
        def _serialize_doc(doc: Document) -> str:
            return json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})
            
        def _deserialize_doc(b) -> Document:
            s = b.decode("utf-8") if isinstance(b, bytes) else b
            data = json.loads(s)
            return Document(page_content=data["page_content"], metadata=data["metadata"])
            
        underlying_store = RedisStore(redis_url=REDIS_URL)
        store = EncoderBackedStore(
            store=underlying_store,
            key_encoder=lambda x: x,
            value_serializer=_serialize_doc,
            value_deserializer=_deserialize_doc
        )

        # [설명: PDR 원리 4단계 - MultiVectorRetriever 연동 및 데이터 적재]
        # 검색용 (Chroma) + 원본용 (Redis) 두 저장소를 하나로 이어주는 리트리버 객체입니다.
        retriever = MultiVectorRetriever(
            vectorstore=vector_db,
            docstore=store,
            id_key=id_key,
        )
        
        print("\n💡 [분배 시작] 작은 조각은 Chroma에, 큰 조각은 Redis에 매핑합니다...")
        
        # 1. 꼬리표가 달린 검색용 조각들을 ChromaDB에 적재합니다.
        retriever.vectorstore.add_documents(sub_docs)
        # 2. 꼬리표의 원본 주인인 큰 문서 통째를 Redis에 적재합니다.
        retriever.docstore.mset(list(zip(doc_ids, all_documents)))
        
        print(f"✅ ChromaDB 저장 완료: {len(sub_docs)}개의 검색용 Child Chunks")
        print(f"✅ Redis 보호 저장 완료: {len(all_documents)}개의 원본 Parent Documents")
        print("🎉 모든 저장 프로세스가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ DB 저장 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()