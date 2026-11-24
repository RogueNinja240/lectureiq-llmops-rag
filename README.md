# LectureIQ — Production LLMOps RAG Assistant

**LectureIQ** is an end-to-end, containerized Retrieval-Augmented Generation (RAG) system built for technical Q&A over noisy, unstructured speech-to-text lecture transcripts.

Unlike clean documentation, raw lecture transcripts contain conversational filler, fragmented explanations, and no section headers. LectureIQ pairs **contextual compression retrieval** and **multi-layer guardrails** with **automated CI/CD evaluation gates**, **online reference-free RAG Triad scoring**, and **dual-layer observability** (Langfuse + Prometheus/Grafana).

---

## Architecture

```mermaid
flowchart LR
    U[User] --> ST[Streamlit Frontend :8501]
    ST -->|/chat stream & /health| API[FastAPI Backend :8000]

    subgraph RAG["LangGraph Agent & Guardrails"]
        API --> IG[Input Guardrails]
        IG --> RET[ChromaDB + Contextual Compression<br/>gpt-5-mini | k=3]
        RET --> GEN[Generation<br/>gpt-5-mini]
        GEN --> OG[Output Guardrails]
    end

    OG -->|Streamed Response| ST
    RAG -.->|Traces & Prompt Versioning| LF[Langfuse Cloud]
    LF <-->|Online RAG Triad Eval<br/>DeepEval gpt-5.4-mini| OE[run_online_eval.py]

    subgraph OBS["Infrastructure Monitoring"]
        NE[Node Exporter :9100] --> PR[Prometheus :9090]
        PR --> GR[Grafana :3000]
    end

    subgraph CICD["Experimentation & CI/CD"]
        ML[MLflow / DagsHub<br/>Champion vs Candidate] --> GHA[GitHub Actions CI/CD]
        GHA --> GHCR[GitHub Container Registry]
    end
