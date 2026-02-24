import os
from dotenv import load_dotenv
import itertools

# .env 파일 로드
load_dotenv()

class Settings:
    PROJECT_NAME: str = "SKT FLY AI Final Project"
    VERSION: str = "1.0.0"

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