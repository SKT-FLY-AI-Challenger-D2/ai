from pydantic import BaseModel 
from typing import Optional, List

class FactResult(BaseModel):
    fake_score: float 
    fake_evidence: List[str]

class DeepfakeResult(BaseModel):
    deepfake_ai_score: float
    deepfake_ai_evidence: List[str]

class LegalResult(BaseModel):
    legal_issue_score: float
    legal_issue_evidence: List[str]


class ModerationState(BaseModel):
    input_text: str = ""
    video_path: str = ""
    audio_path: str = "" # Optional audio path for voice detection
    frame_paths: List[str] = [] # List of paths to extracted frames
    fact: Optional[FactResult] = None 
    deepfake: Optional[DeepfakeResult] = None
    legal: Optional[LegalResult] = None
    is_ad: bool = False
    final_score: float = None 
    report: str = "" 

# class LegalResult(BaseModel):
#     is_falsehood: bool 
#     is_fraud: bool 
#     is_illegal: bool 
#     confidence: float 

# class DetectionResult(BaseModel):
#     is_deepfake: bool
#     is_ai_expert: bool 
#     confidence: float

# class VoiceResult(BaseModel):
#     is_ai_voice: bool
#     confidence: float
#     details: Optional[dict] = None

