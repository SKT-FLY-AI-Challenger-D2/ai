import os
import re
import glob
import sys
from dotenv import load_dotenv

# [중요] 필요한 라이브러리들
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()

# ==========================================
# ⚙️ 설정 (사용자 요청 반영)
# ==========================================
LAW_FOLDER_PATH = "./laws"         # PDF 파일들이 저장된 디렉토리
DB_PATH = "./chroma_db_local"     # 벡터 DB가 저장될 디렉토리

def clean_text(text):
    """법률/판례 텍스트에서 불필요한 노이즈를 제거합니다."""
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
    """
    PDF를 읽어서 법률, 판례, 일반 문서 유형에 맞게 
    지능적으로 텍스트를 분할(Chunking)합니다.
    """
    file_name = os.path.basename(file_path)
    source_name = os.path.splitext(file_name)[0]
    print(f"📖 읽는 중: {source_name}...")
    
    loader = PyPDFLoader(file_path)
    try:
        pages = loader.load()
    except Exception as e:
        print(f"   ⚠️ 파일 로드 실패 ({file_name}): {e}")
        return []
    
    full_text = ""
    for page in pages:
        full_text += clean_text(page.page_content) + "\n"

    docs = []

    # 1. 법률 모드 판별 (제 O조 패턴이 많은 경우)
    article_matches = re.findall(r'(제\s?\d+\s?조\s?\([^)]+\))', full_text)
    # 판례 모드 판별 (주문 및 이유 키워드가 있는 경우)
    is_precedent = re.search(r'주\s*문', full_text) and re.search(r'이\s*유', full_text)

    if len(article_matches) > 5:
        print(f"   👉 [법률 모드] 조항 단위로 분리합니다.")
        article_pattern = r'(제\s?\d+\s?조\s?\([^)]+\))'
        parts = re.split(article_pattern, full_text)
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip()
            num_match = re.search(r'제\s?(\d+)\s?조', header)
            article_no = f"제{num_match.group(1)}조" if num_match else header

            sub_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            sub_chunks = sub_splitter.split_text(content)
            for chunk in sub_chunks:
                tagged_content = f"[{source_name} {header}]\n{chunk}"
                docs.append(Document(
                    page_content=tagged_content, 
                    metadata={"source": source_name, "type": "law", "article": article_no}
                ))

    elif is_precedent:
        print(f"   👉 [판례 모드] 판결문 구조로 분리합니다.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=300, 
            separators=["\n\n", "다만,", "이에 대하여", "살피건대", "."]
        )
        chunks = text_splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            tagged_content = f"[{source_name} 판결문 발췌 {i+1}]\n{chunk}"
            docs.append(Document(
                page_content=tagged_content, 
                metadata={"source": source_name, "type": "precedent", "page": i}
            ))

    else:
        print(f"   👉 [일반/뉴스 모드] 일반 문서로 분리합니다.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            tagged_content = f"[{source_name} 기사/문서 내용]\n{chunk}"
            docs.append(Document(
                page_content=tagged_content, 
                metadata={"source": source_name, "type": "news/etc"}
            ))
            
    return docs

if __name__ == "__main__":
    # 기존 DB가 있다면 충돌 방지를 위해 안내
    if os.path.exists(DB_PATH):
        print(f"⚠️ 기존 DB 폴더('{DB_PATH}')가 발견되었습니다. 최신 데이터를 위해 삭제 후 재생성을 추천합니다.")

    # PDF 파일 목록 가져오기
    pdf_files = glob.glob(os.path.join(LAW_FOLDER_PATH, "*.pdf"))
    if not pdf_files:
        print(f"❌ '{LAW_FOLDER_PATH}' 폴더가 비어있거나 없습니다. PDF를 넣어주세요.")
        exit()

    all_documents = []
    print(f"📂 '{LAW_FOLDER_PATH}' 내 {len(pdf_files)}개의 파일을 처리합니다.")
    
    for pdf_file in pdf_files:
        docs = process_pdf(pdf_file)
        all_documents.extend(docs)
        print(f"   ✅ {len(docs)}개 조각 생성")

    # [모델 설정] 한국어 특화 sroberta 모델 사용 (맥북 MPS 가속)
    print("\n⚙️ 로컬 임베딩 모델 로딩 (jhgan/ko-sroberta-multitask)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'mps'}, # M1/M2/M3 맥북 하드웨어 가속
        encode_kwargs={'normalize_embeddings': True}
    )

    if all_documents:
        print(f"\n🚀 총 {len(all_documents)}개의 법률 조각을 '{DB_PATH}'에 저장합니다...")
        vector_db = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print("🎉 벡터 DB 구축 완료! 이제 AI 수사관이 법을 읽을 수 있습니다.")
    else:
        print("⚠️ 생성된 데이터 조각이 없습니다. PDF 내용을 확인해주세요.")