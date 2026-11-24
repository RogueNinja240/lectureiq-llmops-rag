# LectureIQ — Production LLMOps RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) service for technical Q&A over long-form speech-to-text lecture transcripts, demonstrating a complete LLMOps lifecycle: three-layer guardrails, externalized prompt management, synthetic golden-set generation, LLM-as-judge offline evaluation, statistical regression/promotion gates, online RAG Triad scoring, and dual-layer observability (Langfuse + Prometheus/Grafana).

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.13-blue.svg">
  <img alt="Package manager: uv" src="https://img.shields.io/badge/deps-uv-DE5FE9.svg">
  <img alt="Orchestration: LangGraph" src="https://img.shields.io/badge/orchestration-LangGraph-1C3C3C.svg">
  <img alt="API: FastAPI" src="https://img.shields.io/badge/api-FastAPI-009688.svg">
  <img alt="Eval: DeepEval" src="https://img.shields.io/badge/eval-DeepEval-6E56CF.svg">
  <img alt="Tracking: MLflow" src="https://img.shields.io/badge/tracking-MLflow%20%2F%20DagsHub-0194E2.svg">
  <img alt="Observability: Langfuse + Prometheus + Grafana" src="https://img.shields.io/badge/observability-Langfuse%20%7C%20Prometheus%20%7C%20Grafana-F46800.svg">
  <img alt="CI/CD: GitHub Actions + GHCR" src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20%2B%20GHCR-2088FF.svg">
</p>

---

## Overview

Raw lecture transcripts contain conversational filler, fragmented sentences, and no structural headings—making naive RAG prone to noisy retrieval and hallucinations. LectureIQ runs retrieval, contextual compression, and generation as an explicit **LangGraph** state machine wrapped in three validation gates (`input_guardrail`, `retrieval_guardrail`, and `output_guardrail`).

Around the runtime sits an automated evaluation and observability loop: every configuration change is scored offline against a synthetic golden dataset in **MLflow (DagsHub)**, gated by noise- and drift-aware `pytest` regression/promotion checks, built into multi-stage Docker images published to **GHCR**, and continuously monitored via **Langfuse online evals** and a local **Prometheus + Grafana** stack.

### Key Highlights

- **Graph-based RAG with Contextual Compression** — discrete, testable LangGraph nodes combining ChromaDB (`text-embedding-3-small`, 1024 dims) with `gpt-5-mini` contextual compression to strip transcript filler at query time.
- **Three-layer Guardrails** — jailbreak, PII, and topic validation on user input; prompt-injection checks on retrieved context; relevancy and reading-time checks on generated output—routing cleanly to `error_fallback` (`exception`) or `soft_fallback` (`refrain`).
- **Externalized Prompt Management** — system prompts are versioned and label-promoted in Langfuse, fetched dynamically via `params.yaml` (`prompt_label`), and linked back to every generation trace.
- **Offline Synthetic Evaluation & Statistical Gates** — DeepEval `Synthesizer` (`gpt-5.4-mini`) generates evolved Q&A pairs scored across 7 RAG + `GEval` metrics; `test_regression.py` and `test_promotion.py` enforce noise- and historical-variance thresholds before promoting a challenger.
- **Online Reference-Free RAG Triad** — `src/evals/online_evals/run_online_eval.py` incrementally evaluates live `production` Langfuse traces via a UTC checkpoint and pushes idempotent `answer_relevancy`, `faithfulness`, and `contextual_relevancy` scores back to Langfuse.
- **Dual-Layer Observability** — AI quality, token cost, guardrail trip rates, and latency tracked in **Langfuse**; container host CPU, memory, disk, and network metrics scraped by **Prometheus + Node Exporter** and visualized in **Grafana** inside the same Docker Compose stack.
- **Idempotent Ingestion & Automated GHCR CD** — content-hashed chunks re-embed only modified documents; GitHub Actions builds and publishes API and frontend images to **GHCR (`ghcr.io`)**, with optional cloud deployment cleanly gated behind `vars.ENABLE_AZURE_DEPLOY == 'true'`.

---

## Architecture

```mermaid
flowchart LR
    U["User"] --> ST["Streamlit UI :8501"]
    ST -->|"/chat stream & /health"| API["FastAPI Backend :8000"]

    subgraph RAG["LangGraph Guardrailed RAG"]
        API --> IG["input_guardrail"]
        IG --> RET["retrieve<br/>ChromaDB + Compression, k=3"]
        RET --> RG["retrieval_guardrail"]
        RG --> AUG["augmentation + generation<br/>gpt-5-mini"]
        AUG --> OG["output_guardrail"]
    end

    OG -->|"Streamed Answer / Fallback"| ST
    RAG -.->|"Traces, Guardrail Spans & Prompts"| LF["Langfuse Cloud"]
    LF <-->|"Online RAG Triad<br/>DeepEval gpt-5.4-mini"| OE["run_online_eval.py"]

    subgraph OBS["Dockerized Infrastructure Monitoring"]
        NE["Node Exporter :9100"] --> PR["Prometheus :9090"]
        PR --> GR["Grafana :3000"]
    end

    subgraph CICD["Experimentation & CI/CD"]
        ML["MLflow / DagsHub<br/>Champion vs Challenger"] --> GHA["GitHub Actions CI/CD"]
        GHA --> GHCR["GitHub Container Registry"]
    end
