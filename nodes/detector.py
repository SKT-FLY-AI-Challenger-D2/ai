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
        제공된 20장의 프레임을 분석하여 딥페이크 또는 고도로 정교한 '가상 인간' 여부를 판별하라.

        [1. 핵심 판별 지침: 가짜의 '정적' 특징 탐지]
        - 텍스처 고착화(Texture Fixation): 웃거나 말할 때 주름이나 잡티의 위치가 피부 움직임에 따라 유동적으로 변하지 않고, 마치 스티커처럼 특정 좌표에 고정되어 있는지 확인하라.
        - 안구 및 구강의 깊이감: 가상 인간은 눈동자의 투명도와 구강 내부(치아 뒤쪽 공간)의 어둠이 평면적으로 렌더링되는 경향이 있다. 공간적 깊이감이 물리적으로 타당한지 분석하라.
        - 인위적 노이즈: 실제 잡티와 구별되는 '반복적인 디지털 패턴'의 잡티나, 피부 노이즈가 안면의 굴곡을 무시하고 평면적으로 씌워져 있는지 포착하라.

        [2. 실제 인물 판정 지침: '동적' 불완전성]
        - 동적 생리 현상: 단순히 주름이 있는 것이 아니라, 표정에 따라 근육이 수축하며 주름의 깊이가 실시간으로 변하고, 혈류 변화로 인해 피부색이 미세하게 바뀌는 '동적 변화'가 있다면 실제 인물이다.
        - 외부 상호작용: 안경 테와 닿는 부위의 피부 눌림, 손가락이 얼굴을 스칠 때의 미세한 그림자 왜곡 등 '물체 간 물리적 상호작용'이 완벽하다면 실제 인물로 간주하라.

        [3. 응답 규칙]
        1. 반드시 아래 JSON 형식으로만 응답하라.
        2. 'evidence'에는 외형적 특징이 아닌 '물리적/동적 불일치' 관점의 증거 2가지를 기술하라.

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