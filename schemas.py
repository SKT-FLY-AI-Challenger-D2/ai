from pydantic import BaseModel 
from typing import Optional 

class LegalResult(BaseModel):
    is_falsehood: bool 
    is_fraud: bool 
    is_illegal: bool 
    confidence: float 

class DetectionResult(BaseModel):
    is_deepfake: bool
    is_ai_expert: bool 
    confidence: float

class VoiceResult(BaseModel):
    is_ai_voice: bool
    confidence: float
    details: Optional[dict] = None

class ModerationState(BaseModel):
    input_text: str = ""
    video_path: str = ""
    audio_path: str = "" # Optional audio path for voice detection
    legal: Optional[LegalResult] = None
    deepfake: Optional[DetectionResult] = None
    voice: Optional[VoiceResult] = None
    report: str = "" 