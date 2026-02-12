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

# 환경 변수 로드
load_dotenv()

LAW_FOLDER_PATH = "./laws"
# Docker 서버 설정 (8000번 포트)
# 기존 하드코딩된 부분을 아래와 같이 변경
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8002))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "legal_documents")

class RateLimitedGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            embeddings.append(self.embed_query(t))
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
                    metadata={"source": source_name, "type": "law"}
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
                metadata={"source": source_name, "type": "precedent"}
            ))
    else:
        print("👉 [일반 문서 모드]")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(full_text)
        for chunk in chunks:
            docs.append(Document(
                page_content=f"[{source_name} 문서]\n{chunk}",
                metadata={"source": source_name, "type": "general"}
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

    print(f"\n📦 총 {len(all_documents)}개 문서 조각")

    print("⚙️ Gemini Embedding 로딩...")
    embeddings = RateLimitedGeminiEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    print(f"🚀 Docker Chroma DB 서버({CHROMA_HOST}:{CHROMA_PORT})에 연결 및 저장 중...")
    
    try:
        # 8000번 포트로 떠 있는 Docker 서버에 접속하는 클라이언트 생성
        http_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # client= 파라미터로 명시적으로 전달하여 AttributeError 방지
        vector_db = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            client=http_client
        )
        print("🎉 Docker 서버 저장 완료 (Port: 8000)")
        
    except Exception as e:
        print(f"❌ DB 저장 중 오류 발생: {e}")