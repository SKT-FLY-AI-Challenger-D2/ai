from schemas import ModerationState

def reporter_node(state: ModerationState) -> dict:
    """
    Synthesizes results into a report based on the new schema.
    """
    print("--- Reporter Node ---")
    
    legal = state.legal
    deepfake = state.deepfake
    fact = state.fact
    
    # Calculate final score (simple average for now, or max)
    # If any score is high, the final risk score should be high.
    scores = [
        legal.legal_issue_score if legal else 0.0,
        deepfake.deepfake_ai_score if deepfake else 0.0,
        fact.fake_score if fact else 0.0
    ]
    final_score = sum(scores) / len(scores) if scores else 0.0
    
    report = f"""
# Moderation Analysis Report

## Summary
- **Final Risk Score**: {final_score:.2f} / 1.0

## Detailed Analysis

### 1. Legal Issues
- **Score**: {legal.legal_issue_score if legal else 0.0:.2f}
- **Evidence**:
{chr(10).join([f"  - {e}" for e in legal.legal_issue_evidence]) if legal and legal.legal_issue_evidence else "  - N/A"}

### 2. Deepfake & AI Detection
- **Score**: {deepfake.deepfake_ai_score if deepfake else 0.0:.2f}
- **Evidence**:
{chr(10).join([f"  - {e}" for e in deepfake.deepfake_ai_evidence]) if deepfake and deepfake.deepfake_ai_evidence else "  - N/A"}

### 3. Fact Check
- **Score**: {fact.fake_score if fact else 0.0:.2f}
- **Evidence**:
{chr(10).join([f"  - {e}" for e in fact.fake_evidence]) if fact and fact.fake_evidence else "  - N/A"}

## Conclusion
{'⚠️ High risk content detected.' if final_score > 0.7 else '✅ Content appears relatively safe, but review evidence.'}
"""
    return {"report": report, "final_score": final_score}
