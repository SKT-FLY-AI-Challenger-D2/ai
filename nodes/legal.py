import google.generativeai as genai
import os
import json
from schemas import ModerationState, LegalResult

def legal_node(state: ModerationState) -> dict:
    """
    Analyzes input text for legal issues using Gemini.
    """
    print(f"--- Legal Node ---\nProcessing text: {state.input_text}")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
         print("Warning: GOOGLE_API_KEY not found. Returning dummy result.")
         return {"legal": LegalResult(is_falsehood=False, is_fraud=False, is_illegal=False, confidence=0.0)}

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Analyze the following text for legal issues, specifically fraud, illegal activities, and falsehoods.
        Text: "{state.input_text}"
        
        Return a JSON object with the following keys:
        - is_illegal: bool
        - is_fraud: bool
        - is_falsehood: bool
        - confidence: float (0.0 to 1.0)
        """
        
        response = model.generate_content(prompt)
        text_resp = response.text.strip()
        
        # Extract JSON
        if "```json" in text_resp:
            json_str = text_resp.split("```json")[1].split("```")[0]
        elif "{" in text_resp:
            json_str = "{" + text_resp.split("{", 1)[1].rsplit("}", 1)[0] + "}"
        else:
            json_str = text_resp
            
        data = json.loads(json_str)
        
        result = LegalResult(
            is_falsehood=data.get("is_falsehood", False),
            is_fraud=data.get("is_fraud", False),
            is_illegal=data.get("is_illegal", False),
            confidence=data.get("confidence", 0.0)
        )
        
        return {"legal": result}

    except Exception as e:
        print(f"Error in legal_node: {e}")
        return {"legal": LegalResult(is_falsehood=False, is_fraud=False, is_illegal=False, confidence=0.0)}
