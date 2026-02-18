<div align="center">

# Deep Research Agent

### Autonomous, Hierarchical Research System with Verifiable Provenance

[![Kestra](https://img.shields.io/badge/Orchestration-Kestra-6366f1?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0id2hpdGUiLz48L3N2Zz4=)](https://kestra.io)
[![Supabase](https://img.shields.io/badge/Memory-Supabase-3ecf8e?style=flat-square&logo=supabase)](https://supabase.com)
[![Mem0](https://img.shields.io/badge/Semantic-Mem0-purple?style=flat-square)](https://mem0.ai)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?style=flat-square&logo=google)](https://deepmind.google/technologies/gemini/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Built%20for-Hackathon-D22128?style=flat-square)](https://hackathon.dev)

</div>

---

## Demo Video: 

Click the image above to watch the demo video

[![Deep Research Agent](https://img.youtube.com/vi/ABoEuW0wUTQ/maxresdefault.jpg)](https://youtu.be/ABoEuW0wUTQ)

---

## The Problem
In an era of information overload, finding high-quality, verified answers to complex questions is difficult. Simple RAG (Retrieval-Augmented Generation) systems often fail on multi-step reasoning tasks and suffer from:
1.  **Hallucination**: Inventing facts when sources are missing.
2.  **Shallow Analysis**: Failing to connect dots across disparate sources.
3.  **Lack of Provenance**: Giving answers without traceable evidence.

## The Solution: Deep Research 
The **Deep Research Agent** is an autonomous system designed to emulate a human researcher. It doesn't just search; it **plans, investigates, validates, and synthesizes**.

### Core Methodology
1.  **Hierarchical Planning**: Instead of a linear chain of thought, the agent builds a dynamic **Task Graph**. It breaks complex goals into sub-tasks (research, validation, extraction) and executes them in parallel.
2.  **Evidence-First**: Every claim in the final report is backed by an **Evidence Graph**. This graph links specific text snippets from sources to the claims they support (or contradict).
3.  **Self-Correction**: A **Reflexion Loop** monitors progress. If the gathered information is insufficient or contradictory, the agent dynamically **re-plans**, adding new tasks to fill gaps.

---

## Key Features

- **Hierarchical Task Graph** - Dynamic planning with dependency management and parallel execution.
- **Evidence Graph** - Traceable claim provenance linking every sentence to its source.
- **Persistent Memory** - Long-term storage using Supabase (pgvector) and semantic recall via Mem0.
- **Source Validation** - "LLM-as-a-Judge" evaluates credibility, domain authority, and bias.
- **Reflexion & Re-planning** - Autonomous quality control that modifies the plan at runtime.
- **Multi-Hop Retrieval** - Recursively follows citations to find primary sources.
- **Adaptive Model Routing** - Routes simple tasks to faster models and complex reasoning to stronger models.

---

## Architecture

```mermaid
flowchart TB
    User[User / UI] --> |"Query"| HPA[Hierarchical Planner]

    subgraph Memory["Memory & Storage"]
        Supabase[(Supabase PG)]
        Mem0[Mem0 Semantic]
    end

    subgraph Planning["Pre-Computation Phase"]
        HPA --> |"Decompose"| TG[Task Graph]
    end

    subgraph Execution["Runtime Execution Engine"]
        TG --> |"Dispatch"| Executor[Task Executor]
        
        Executor --> |"Parallel"| Agents
        
        subgraph Agents["Agent Swarm"]
            WSA[Web Search]
            ASA[Academic Search]
            TSA[Technical Search]
            CC[Citation Crawler]
        end
    end

    subgraph Analysis["Analysis & Synthesis"]
        Agents --> |"Raw Data"| Val[Source Validator]
        Val --> |"Validated"| CE[Claim Extractor]
        CE --> |"Claims"| EG[Evidence Graph]
        
        EG --> |"Provneance"| Ref[Reflexion Agent]
        Ref --> |"Critique"| HPA
        
        EG --> |"Final Graph"| Syn[Synthesizer]
    end

    Syn --> |"Report + Json"| User
    
    %% Connections to Memory
    HPA & Agents & CE -.-> Memory

    style HPA fill:#6366f1,stroke:#4338ca,color:#fff
    style TG fill:#8b5cf6,stroke:#7c3aed,color:#fff
    style EG fill:#10b981,stroke:#059669,color:#fff
    style Memory fill:#f59e0b,stroke:#d97706,color:#fff
```

---

## Project Structure

```
Deep-Research-Agent/
├── src/
│   ├── planning/            # Task Graph & Execution Engine
│   │   ├── hierarchical_planner.py
│   │   ├── task_graph.py
│   │   └── executor.py
│   ├── evidence/            # Evidence Graph Logic
│   │   └── graph.py
│   ├── memory/              # Persistence Layer
│   │   ├── memory_api.py
│   │   └── supabase_store.py
│   ├── agents/              # Specialized Agents
│   │   ├── master_planner.py
│   │   ├── source_validator.py
│   │   ├── claiming_extractor.py
│   │   └── ...
│   └── core/                # Core Infrastructure
│       └── llm_client.py    # Adaptive Model Router
├── prompts/                 # System Prompts (.md)
├── kestra/                  # Orchestration Workflows
├── main.py                  # V2 Orchestrator Entrypoint
└── server.py                # REST API Server
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Supabase](https://supabase.com) Account (for memory)
- API Keys: Google Gemini (recommended), or Together AI / OpenRouter

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Deep-Research-Agent.git
   cd Deep-Research-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your keys:
   ```ini
   GEMINI_API_KEY=your_key_here
   SUPABASE_URL=your_url
   SUPABASE_KEY=your_key
   ```

### Running the Agent

**Option 1: CLI Mode**
```bash
python main.py
```
This runs a sample research query defined in `main.py`.

**Option 2: API Server**
```bash
python server.py
```
Starts a REST API at `http://localhost:8000`.

---

## Deployment

- **Backend**: Deploy as a Docker container on specialized GPU clouds or standard PaaS (Railway, Render).
- **Frontend**: Next.js application deployable on Vercel.
- **Database**: Managed Supabase instance.


