from schemas import ModerationState

def reporter_node(state: ModerationState) -> dict:
    """
    Synthesizes results into a report.
    """
    print("--- Reporter Node ---")
    
    legal = state.legal
    deepfake = state.deepfake
    voice = state.voice
    
    report = f"""
# Moderation Report

## Input Data
- Text: {state.input_text}
- Video: {state.video_path}
- Audio: {state.audio_path}

## Legal Analysis
- Illegal: {legal.is_illegal}
- Fraud: {legal.is_fraud}
- Falsehood: {legal.is_falsehood}
- Confidence: {legal.confidence}

## Deepfake Detection
- Deepfake: {deepfake.is_deepfake}
- AI Expert: {deepfake.is_ai_expert}
- Confidence: {deepfake.confidence}

## Voice Analysis
- AI Voice: {voice.is_ai_voice if voice else 'N/A'}
- Confidence: {voice.confidence if voice else 0.0}

## Summary
The content has been analyzed. 
{'This content contains potential legal issues.' if legal.is_illegal or legal.is_fraud else 'No specific legal issues detected.'}
{'This video appears to be a deepfake.' if deepfake.is_deepfake else 'No deepfake detected.'}
{'This audio appears to be AI-generated.' if voice and voice.is_ai_voice else 'No AI voice detected.'}
"""
    return {"report": report}
