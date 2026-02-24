import cv2
import os
import json
import numpy as np
import base64
import requests
import time
from google import genai
from google.genai import types
from schemas import ModerationState, DeepfakeResult
from google.genai.errors import APIError

from config import settings

# 1. 얼굴 인식기 초기화 (OpenCV 기본 모델 사용)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_start_time(video_path):
    """영상 내에서 얼굴이 처음 감지되는 타임스탬프 탐색"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret: break
        frame_count += 1
        # 성능을 위해 5프레임 단위로 탐색
        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                video.release()
                return frame_count / fps
    video.release()
    return 0

def extract_cropped_face_frames(video_path, start_time, count=20):
    """얼굴 감지 시점부터 3초간 20장의 고해상도 얼굴 크롭 이미지 추출"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    
    # 분석 구간 설정 (시작점부터 3초간)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + 3) * fps)
    interval = max(1, (end_frame - start_frame) // count)
    
    image_parts = []

    for i in range(count):
        target_pos = start_frame + (i * interval)
        video.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        ret, frame = video.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # 가장 큰 얼굴 선택 및 마진 추가 크롭
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            margin = int(w * 0.2)
            y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
            
            face_img = frame[y1:y2, x1:x2]
            # 모델 분석 최적화 해상도 (안정성을 위해 1024 유지)
            face_img = cv2.resize(face_img, (1024, 1024))
            
            _, buffer = cv2.imencode('.jpg', face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            image_parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
        else:
            # 얼굴 미감지 시 원본 프레임 전송
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            image_parts.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
            
    video.release()
    return image_parts

# url 수정 필요 
RUNPOD_URL = os.getenv(
    "RUNPOD_URL",
    "https://utkffbgubk59n7-8000.proxy.runpod.net/inference"
)

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def invoke_custom_model(image_bytes: bytes) -> dict:
    """자체 Deepfake 탐지 모델(RunPod) 호출 (영어 프롬프트/응답)"""
    prompt = "<image>\nIs this image fake or real?\nASSISTANT:"

    try:
        if not RUNPOD_API_KEY:
            return {"status": "error", "message": "RUNPOD_API_KEY not set"}

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "prompt": prompt,
            "image_base64": img_b64
        }

        headers = {
            "x-api-key": RUNPOD_API_KEY,
            "Content-Type": "application/json"
        }

        print(f"[Detector] 자체 모델 호출 시도: {RUNPOD_URL}")

        response = requests.post(
            RUNPOD_URL,
            json=payload,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            res_content = response.json().get("result", "").lower()
            is_fake = "fake" in res_content
            print(f"[Detector] 자체 모델 응답 수신 성공: {res_content}")
            return {
                "status": "success",
                "is_fake": is_fake,
                "raw": res_content
            }
        else:
            print(f"[Detector] 자체 모델 호출 실패 (Status: {response.status_code})")
            return {
                "status": "error",
                "message": f"API Error: {response.status_code}",
                "detail": response.text
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def detector_node(state: ModerationState) -> dict:
    """
    고해상도 얼굴 포렌식 분석 노드
    최종 결과에서 score와 2가지 핵심 증거(evidence)만 추출함.
    """
    start_time_str = time.strftime("%H:%M:%S")
    print(f"--- [Detector Node] 시작 시각: {start_time_str} ---")
    print(f"대상 파일: {state.video_path}")
    
    if not state.video_path or not os.path.exists(state.video_path):
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["영상 파일 없음"])}

    try:
        # 1. 얼굴 탐지 및 프레임 추출
        start_pt = get_face_start_time(state.video_path)
        face_frames = extract_cropped_face_frames(state.video_path, start_time=start_pt, count=20)
        
        if not face_frames:
            return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["얼굴 감지 실패"])}

        # 2. 자체 모델(High Precision Gate) 호출 - 단일 프레임 사용
        single_face_bytes = face_frames[0].inline_data.data
        custom_result = invoke_custom_model(single_face_bytes)
        print(f"[Custom Model] Result: {custom_result}")

        # Gemini 클라이언트 설정
        api_key = os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # 3. 분기에 따른 프롬프트 설정
        is_custom_fake = custom_result.get("is_fake")
        
        if is_custom_fake:
            # [CASE 1] 자체 모델이 Fake라고 판정한 경우 (정밀 확인 및 증거 수집)
            print("--- [Detector] 자체 모델 Fake 판정 -> Gemini 정밀 확인 모드 ---")
            prompt = """
            [역할: 최고 등급 디지털 영상 포렌식 수사관]
            전단계 분석 모델이 이 영상을 'FAKE'로 판정했다. 
            제공된 20장의 프레임을 분석하여 이 판정을 검증하고, 구체적인 시각적 증거(아티팩트) 2가지를 찾아라.

            [분석 중점]
            - 얼굴 경계면의 부자연스러운 계단 현상이나 블렌딩 오류.
            - 눈동자 반사광의 비일관성 또는 구강 내부의 평면적 질감.
            - 표정 변화 시 특정 피부 영역의 텍스처 고착 현상.

            [응답 규칙]
            1. 반드시 아래 JSON 형식으로만 응답하라.
            2. 'evidence'에는 판정을 뒷받침하는 구체적인 물리적 결함 2가지를 기술하라.
            3. 'score'는 0.5(의심) ~ 1.0(확신) 사이로 책정하라. (1.0에 가까울수록 확실한 가짜임)
            4. 특정 프레임을 언급하지는 말 것. 

            {
            "score": (0.5~1.0),
            "evidence": ["첫 번째 증거", "두 번째 증거"]
            }
            """
        else:
            # [CASE 2] 자체 모델이 Real이라고 하거나 에러인 경우 (독자적인 표준 분석 수행)
            print("--- [Detector] 자체 모델 Real 판정(혹은 에러) -> Gemini 표준 분석 모드 ---")
            prompt = """
            [역할: 최고 등급 디지털 영상 포렌식 수사관]
            제공된 20장의 프레임을 분석하여 딥페이크 또는 고도로 정교한 '가상 인간' 여부를 판별하라.

            [1. 핵심 판별 지침: 가짜의 '정적' 특징 탐지]
            - 텍스처 고착화(Texture Fixation): 웃거나 말할 때 주름이나 잡티의 위치가 피부 움직임에 따라 유동적으로 변하지 않고, 마치 스티커처럼 특정 좌표에 고정되어 있는지 확인하라.
            - 안구 및 구강의 깊이감: 가상 인간은 눈동자의 투명도와 구강 내부의 어둠이 평면적으로 렌더링되는 경향이 있다. 
            - 인위적 노이즈: 실제 잡티와 구별되는 '반복적인 디지털 패턴'의 잡티 탐지.

            [2. 실제 인물 판정 지침: '동적' 불완전성]
            - 동적 생리 현상: 근육이 수축하며 주름의 깊이가 실시간으로 변하고 피부색이 미세하게 바뀌는 '동적 변화' 확인.
            - 외부 상호작용: 물체 간 물리적 상호작용(그림자, 피부 눌림)의 완벽함 확인.

            [3. 응답 규칙]
            1. 반드시 아래 JSON 형식으로만 응답하라.
            2. 'evidence'에는 외형적 특징이 아닌 '물리적/동적 불일치' 관점의 증거 2가지를 기술하라.
            3. 'score'는 0.0(완전 실제) ~ 1.0(완전 가짜) 사이로 책정하라. (1.0에 가까울수록 딥페이크일 확률이 높음)
            4. 특정 프레임을 언급하지는 말 것. 

            {
            "score": (0.0~1.0),
            "evidence": ["첫 번째 증거", "두 번째 증거"]
            }
            """

        # Gemini 호출
        for model_name in settings.MODELS:
            print(f"[Detector] Gemini 호출 시도 (Model: {client.models.get_model('gemini-3-flash-preview').name if hasattr(client.models, 'get_model') else 'gemini-3-flash-preview'})")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=face_frames + [prompt],
                    config=types.GenerateContentConfig(temperature=0.1, top_p=0.85)
                )
                print("[Detector] Gemini 응답 수신 성공")
                print("Gemini Response: ", response.text)

                # 4. 결과 파싱
                res_text = response.text.strip()
                if "```json" in res_text:
                    res_text = res_text.split("```json")[1].split("```")[0].strip()
                elif "{" in res_text:
                    res_text = "{" + res_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                
                data = json.loads(res_text)
                
                result = DeepfakeResult(
                    deepfake_ai_score=float(data.get("score", 0.0)),
                    deepfake_ai_evidence=data.get("evidence", [])[:2]
                )

                end_time_str = time.strftime("%H:%M:%S")
                print(f"--- [Detector Node] 종료 시각: {end_time_str} ---")
                return {"deepfake": result}
            except APIError as e:
                print(f"{model_name} API 에러(트래픽 등): {e}. 다음 모델 시도.")
                continue

    except Exception as e:
        print(f"Error in detector_node: {e}")
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=[f"에러: {str(e)}"])}