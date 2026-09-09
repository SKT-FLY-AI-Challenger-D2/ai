import os
from dotenv import load_dotenv
import itertools

# .env 파일 로드
load_dotenv()

class Settings:
    PROJECT_NAME: str = "SKT FLY AI Final Project"
    VERSION: str = "1.0.0"

    # YouTube 다운로드용 쿠키 파일 경로 (TASK-07). 값이 없으면 이 저장소 루트의
    # youtube_cookies.txt를 기본값으로 쓴다(개인 절대경로 하드코딩 제거).
    YOUTUBE_COOKIE_PATH: str = os.getenv(
        "YOUTUBE_COOKIE_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_cookies.txt"),
    )

    # PO 토큰은 선택값이다. 기존 하드코딩 토큰은 재가동 시점 실제 호출에서
    # 포맷을 얻지 못해 사용 불가로 확인됐다. 비워두면 공개 영상 경로로만
    # 동작하고, 로그인 전용 등 새 토큰이 필요한 대상에서만 조건부로 채운다.
    YOUTUBE_PO_TOKEN: str = os.getenv("YOUTUBE_PO_TOKEN", "")

    # YouTube가 요구하는 PO 토큰(BotGuard attestation)을 자동 발급해주는 bgutil
    # 제공자 서버 주소 (TASK-16). Compose 내부 DNS. 비우면 플러그인이 127.0.0.1:4416을 본다.
    BGUTIL_POT_BASE_URL: str = os.getenv("BGUTIL_POT_BASE_URL", "")

    # YouTube 요청을 데이터센터가 아닌 IP로 우회할 프록시 (TASK-16). 예:
    # http://user:pass@proxy-host:port. Azure 등 클라우드 IP는 YouTube가
    # 봇 의심으로 자주 차단하는데, 주거용 프록시를 쓰면 익명 + PO 토큰만으로도
    # 안정적으로 받을 수 있어 개인 계정 쿠키를 쓰지 않아도 된다.
    YOUTUBE_PROXY: str = os.getenv("YOUTUBE_PROXY", "")

    # --- LLM / 외부 API (VAL-0144: 이전엔 각 노드가 직접 os.getenv로 읽어 설정
    #     검증·누락 진단이 분산됐다. config.py로 일원화) ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")      # 없으면 None
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")      # 없으면 None (fact_check 검색)

    # --- ChromaDB (법률 RAG) ---
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8002"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "legal_documents")

    # --- Redis (laws_embedding.py 재임베딩 시에만, --profile reembed) ---
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    MODELS = []#= [os.getenv(f"YOUTUBE_API_KEY{i}") for i in range(10)]
    for i in range(10):
        key = os.getenv(f"MODEL_NAME{i}", "")
        if key != "":
            MODELS.append(key)

    # model_name_cycle = itertools.cycle(models)

    # def get_next_model_name(self) -> str:
    #     """다음 순서의 API 키를 반환합니다."""
    #     key = self.model_name_cycle
    #     print(f"[Config] Model 이름 변경: {key}")
    #     return next(key)



settings = Settings()