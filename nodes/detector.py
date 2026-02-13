import os
import json
import base64
import requests
import re
from schemas import ModerationState, DeepfakeResult

# 현재 모델은 json 출력이 정상적으로 안되는 듯함 

def detector_node(state: ModerationState) -> dict:
    """
    Analyzes extracted frames for deepfakes using a custom RunPod LMM API.
    Returns DeepfakeResult.
    """
    print(f"--- Detector Node ---\nProcessing {len(state.frame_paths)} frames for video: {state.video_path}")
    
    if not state.frame_paths:
        print("No frame paths provided for detection.")
        return {"deepfake": DeepfakeResult(deepfake_ai_score=0.0, deepfake_ai_evidence=["No frames provided for analysis."])}

    URL = "https://n9l19dq722i265-8000.proxy.runpod.net/inference"
    manipulation_evidence = []
    total_score = 0.0
    processed_frames = 0

    for i, frame_path in enumerate(state.frame_paths):
        if not os.path.exists(frame_path):
            continue

        try:
            print(f"Analyzing frame {i+1}: {frame_path}...")
            with open(frame_path, "rb") as image_file:
                img_b64 = base64.b64encode(image_file.read()).decode('utf-8')

            data = {
                "prompt": """<image>
당신은 딥페이크 및 AI 생성 콘텐츠 분석 전문가입니다. 제공된 이미지를 분석하여 딥페이크나 AI 생성 여부를 판별해주세요.
응답은 반드시 아래의 JSON 형식으로만 작성해야 합니다:

{
  "deepfake_ai_score": 0.0 ~ 1.0 (1.0에 가까울수록 기술적 조작이나 AI 생성 확률이 높음),
  "deepfake_ai_evidence": ["근거 1", "근거 2", ...] (조작의 증거가 되는 시각적 결함이나 특이사항을 짧게 나열)
}

분석 시 다음 사항을 중점적으로 확인하세요:
1. 피부 질감의 부자연스러움
2. 눈, 코, 입 주변의 뭉개짐이나 어색한 연결
3. 배경과의 경계선 왜곡
4. 조명 및 그림자의 불일치

ASSISTANT:""",
                "image_base64": img_b64
            }

            response = requests.post(URL, json=data, timeout=30)
            if response.status_code == 200:
                full_result = response.json().get("result", "")
                
                # Extract response text after Assistant tag
                if "ASSISTANT:" in full_result:
                    result_text = full_result.split("ASSISTANT:")[-1].strip()
                elif "assistant:" in full_result:
                    result_text = full_result.split("assistant:")[-1].strip()
                else:
                    result_text = full_result.strip()

                print(f"Frame {i+1} raw response: {result_text}")

                # JSON Parsing
                try:
                    # Clean the result text: some LLMs escape underscores (e.g., \_)
                    json_str = result_text.replace("\\_", "_")
                    
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "{" in json_str:
                        json_str = "{" + json_str.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                    
                    data_resp = json.loads(json_str)
                    
                    frame_score = float(data_resp.get("deepfake_ai_score", 0.0))
                    frame_evidence = data_resp.get("deepfake_ai_evidence", [])
                    
                    if not isinstance(frame_evidence, list):
                        frame_evidence = [str(frame_evidence)]
                    
                    total_score += frame_score
                    processed_frames += 1

                    # Only collect evidence if manipulation is suspected for this frame
                    if frame_score > 0.5:
                        for ev in frame_evidence:
                            # Clean up common prefixes like "근거 1:", "Frame 1:" etc. if desired
                            clean_ev = re.sub(r'^(근거\s*\d+:|Frame\s*\d+:)\s*', '', ev).strip()
                            if clean_ev:
                                manipulation_evidence.append(clean_ev)
                    
                except Exception as json_e:
                    print(f"JSON parsing error for frame {i+1}: {json_e}")
                    # Fallback if JSON parsing fails but text mentions real/fake
                    lower_text = result_text.lower()
                    if "fake" in lower_text or "ai 생성" in lower_text or "조작" in lower_text:
                        total_score += 0.8
                        manipulation_evidence.append(f"기술적 부자연스러움 및 조작 흔적이 발견되었습니다.")
                        processed_frames += 1
            else:
                print(f"Error calling RunPod API for frame {i+1}: {response.status_code}")

        except Exception as e:
            print(f"Exception during RunPod API call for frame {i+1}: {e}")

    if processed_frames > 0:
        avg_score = total_score / processed_frames
    else:
        avg_score = 0.0

    # Aggregate evidence: Pick only several key evidences from manipulated frames
    final_evidence = []
    if manipulation_evidence:
        # Deduplicate and take top unique ones
        seen = set()
        for ev in manipulation_evidence:
            if ev not in seen:
                final_evidence.append(ev)
                seen.add(ev)
        final_evidence = final_evidence[:4] # Limit to 4 key points

    result = DeepfakeResult(
        deepfake_ai_score=avg_score,
        deepfake_ai_evidence=final_evidence
    )

    print(f"Final Detection Score: {avg_score:.2f}")
    return {"deepfake": result}
