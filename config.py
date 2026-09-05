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