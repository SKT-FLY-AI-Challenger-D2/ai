from langgraph.graph import StateGraph, END, START
from schemas import ModerationState
from nodes import legal_node, detector_node, reporter_node, fact_check_node, ad_check_node

# Define the graph
workflow = StateGraph(ModerationState)

# Add nodes
workflow.add_node("ad_check", ad_check_node)
workflow.add_node("legal", legal_node)
workflow.add_node("fact_check", fact_check_node)
workflow.add_node("detector", detector_node)
workflow.add_node("reporter", reporter_node)

# Define logic for conditional edge
def route_after_ad_check(state: ModerationState):
    if state.is_ad:
        return ["legal", "fact_check", "detector"]
    else:
        return END

# Define edges
workflow.add_edge(START, "ad_check")

# Conditional edge from ad_check
workflow.add_conditional_edges(
    "ad_check",
    route_after_ad_check,
)

# All analysis nodes flow to reporter
workflow.add_edge("legal", "reporter")
workflow.add_edge("fact_check", "reporter")
workflow.add_edge("detector", "reporter")

workflow.add_edge("reporter", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Test the graph with dummy data
    print("Running graph with test data...")
    inputs = {
        "input_text": "This is a potential fraud case involves fake money.",
        "video_path": "test_video.mp4" 
    }
    
    # Run the graph
    result = app.invoke(inputs)
    
    print("\n--- Final Result ---")
    print(result.get("report", "No report generated."))

