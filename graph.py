from langgraph.graph import StateGraph, END, START
from schemas import ModerationState
from nodes import legal_node, detector_node, reporter_node, voice_detector_node

# Define the graph
workflow = StateGraph(ModerationState)

# Add nodes
workflow.add_node("legal", legal_node)
workflow.add_node("detector", detector_node)
workflow.add_node("voice", voice_detector_node)
workflow.add_node("reporter", reporter_node)

# Define edges
workflow.add_edge(START, "legal")
workflow.add_edge(START, "detector")
workflow.add_edge(START, "voice")
workflow.add_edge("legal", "reporter")
workflow.add_edge("detector", "reporter")
workflow.add_edge("voice", "reporter")
workflow.add_edge("reporter", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Test the graph with dummy data
    print("Running graph with test data...")
    inputs = {
        "input_text": "This is a potential fraud case involves fake money.",
        "video_path": "test_video.mp4" # user needs to provide a real path or handle the error
    }
    
    # Run the graph
    result = app.invoke(inputs)
    
    print("\n--- Final Result ---")
    print(result["report"])
    print(f"Legal Fraud Detected: {result['legal'].is_fraud}")
    print(f"Deepfake Detected: {result['deepfake'].is_deepfake}")

