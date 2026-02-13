import os
import json
import base64
import requests
from schemas import ModerationState, DeepfakeResult

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
    all_evidence = []
    fake_evidence_list = []
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
You are an expert in AI-generated content analysis. Follow these steps:
• Extract Common Ground: Identify overlapping details (e.g., shadows, outlines, object alignment) across all three responses.
• Filter Minority Claims: Discard observations mentioned by only one model unless they are critical (e.g.,glaring artifacts).
• Structure Hierarchically: Group explanations by category (e.g., lighting, geometry, textures) for clarity.
• Maintain Original Format: Begin with “This is a [real/fake] image.” followed by a concise, semicolon-separated list of consolidated evidence.
• Avoid Redundancy: Rephrase overlapping points to eliminate repetition while preserving technical accuracy.
• Ensure Logical Consistency: If any response contains nonsensical, contradictory, or infinite loop reasoning, disregard that portion of the answer.
ASSISTANT:""",
                "image_base64": img_b64
            }

            response = requests.post(URL, json=data, timeout=30)
            if response.status_code == 200:
                full_result = response.json().get("result", "")
                
                # Split by ASSISTANT: to get only the response part
                if "ASSISTANT:" in full_result:
                    result_text = full_result.split("ASSISTANT:")[-1].strip()
                elif "assistant:" in full_result:
                    result_text = full_result.split("assistant:")[-1].strip()
                else:
                    result_text = full_result.strip()

                lower_text = result_text.lower()
                print(f"Frame {i+1} result: {lower_text}")
                
                processed_frames += 1
                
                if "real" in lower_text:
                    frame_score = 0.0
                    all_evidence.append(f"Frame {i+1} judged as REAL.")
                elif "fake" in lower_text:
                    frame_score = 1.0
                    evidence_text = f"Frame {i+1} judged as FAKE: {result_text}"
                    fake_evidence_list.append(evidence_text)
                    all_evidence.append(evidence_text)
                else:
                    print(f"Ambiguous result for frame {i+1}: {result_text}")
                    frame_score = 0.5
                    all_evidence.append(f"Frame {i+1} was inconclusive: {result_text}")

                total_score += frame_score
            else:
                print(f"Error calling RunPod API for frame {i+1}: {response.status_code}")

        except Exception as e:
            print(f"Exception during RunPod API call for frame {i+1}: {e}")

    if processed_frames > 0:
        avg_score = total_score / processed_frames
    else:
        avg_score = 0.0

    # Logic: if score >= 0.5, take evidence from one of the fake frames
    if avg_score >= 0.5 and fake_evidence_list:
        final_evidence = [fake_evidence_list[0]]
    else:
        final_evidence = ["딥페이크, 혹은 AI 생성되었다고 보기 힘든 영상입니다."]

    result = DeepfakeResult(
        deepfake_ai_score=avg_score,
        deepfake_ai_evidence=final_evidence
    )

    return {"deepfake": result}
