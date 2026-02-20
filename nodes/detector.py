import cv2
import os
import json
import numpy as np
from google import genai
from google.genai import types
from schemas import ModerationState, DeepfakeResult

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

def extract_cropped_face_frames(video_path, start_time, count=15):
    """얼굴 감지 시점부터 3초간 20장의 고해상도 얼굴 크롭 이미지 추출"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    
    # 분석 구간 설정 (시작점부터 3초간)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + 4) * fps)
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

def detector_node(state: ModerationState) -> dict:
    """
    고해상도 얼굴 포렌식 분석 노드
    최종 결과에서 score와 2가지 핵심 증거(evidence)만 추출함.
    """
    print(f"--- [Detector Node] 정밀 분석 시작 ---\n대상 파일: {state.video_path}")
    
    if not state.video_path or not os.path.exists(state.video_path):
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["영상 파일 없음"])}

    try:
        # 1. 얼굴 탐지 및 프레임 추출 (창현님 로직)
        start_pt = get_face_start_time(state.video_path)
        face_frames = extract_cropped_face_frames(state.video_path, start_time=start_pt)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        
        # 2. 요약된 응답을 위한 강화된 프롬프트
        prompt = """
        [역할: 최고 등급 디지털 영상 포렌식 수사관]
        제공된 15장의 고해상도 연속 프레임(약 4초 구간)을 분석하여 딥페이크(AI 얼굴 교체 및 생성) 여부를 판별하라.

        [중요: 오탐 주의 지침 - 가장 중요한 원칙]
        다음 현상은 AI 조작이 아닌 '환경적 요인'으로 간주하여 score를 보수적으로(낮게) 측정하라. 아래 현상으로 설명될 수 있는 결함이라면 반드시 실제 인물 영상으로 간주한다.
        1. 강한 조명으로 인한 화이트아웃(접시물 현상) 및 피부 질감 상실.
        2. 유튜브 압축 노이즈(Block Artifacts) 및 저화질로 인한 경계선 깨짐, 입술/치아 주변의 단순 뭉개짐.
        3. 뷰티 필터(매끄러운 피부, 얼굴 윤곽 보정) 적용은 인간의 의도적 영상 편집일 뿐 AI 생성물이 아님.

        [판정 지침 - AI 조작의 결정적 증거]
        오직 '생성형 AI' 특유의 시공간적, 물리적 결함에만 집중하라:
        - 눈동자 내부의 비대칭적 반사광 모순.
        - 입술과 치아의 기괴한 물리적 융합 (저화질로 인한 단순 뭉개짐이 아닌 3D 형태학적 파괴).
        - 15프레임 흐름상 배경 및 이목구비의 비정상적인 요동(지터링) 현상.
        조금이라도 모호하거나 환경적 요인으로 설명 가능한 경우 score를 매우 낮게 측정하라.
        
        [응답 규칙]
        1. 반드시 아래 JSON 형식으로만 응답하라.
        2. 'evidence'에는 종합적으로 가장 결정적인 증거 2가지만 각각 한 문장씩, 총 두 문장으로 간단히 기술하라. (만약 환경적 요인으로 인해 실제 영상으로 판정했다면, 그 이유를 증거란에 명시할 것)

        {
        "score": (0.0~1.0 사이),
        "evidence": ["첫 번째 핵심 증거", "두 번째 핵심 증거"]
        }
        """
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=face_frames + [prompt],
            config=types.GenerateContentConfig(temperature=0.1, top_p=0.85)
        )

        # 3. 결과 파싱
        res_text = response.text.strip()
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        elif "{" in res_text:
            res_text = "{" + res_text.split("{", 1)[1].rsplit("}", 1)[0] + "}"
        
        data = json.loads(res_text)
        
        # schemas.py 구조에 맞게 변환 (score와 2개의 문장만 전달)
        result = DeepfakeResult(
            deepfake_ai_score=float(data.get("score", 0.0)),
            deepfake_ai_evidence=data.get("evidence", [])[:2] # 확실히 2개만 추출
        )

        return {"deepfake": result}

    except Exception as e:
        print(f"Error in detector_node: {e}")
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=[f"에러: {str(e)}"])}