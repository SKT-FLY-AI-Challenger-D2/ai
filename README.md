## Multi-Agent 기반 AI 모듈

- 법률 기반 허위/사기 광고 여부
- 딥페이크 광고 여부
- AI 생성 전문가 광고 여부

# System Architecture Overview
## Graph Structure

The system is built using **LangGraph** and follows a parallel execution pattern for analysis nodes, aggregating results in a final reporter node.

```mermaid
graph TD
    START --> Legal[Legal Node]
    START --> Detector[Video Detector Node]
    START --> Voice[Voice Detector Node]
    
    Legal --> Reporter[Reporter Node]
    Detector --> Reporter
    Voice --> Reporter
    
    Reporter --> END
