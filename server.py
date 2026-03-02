from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import uuid
import asyncio
import os
import logging
from datetime import datetime

from main import DeepResearchOrchestratorV2, ResearchResultV2
from src.storage import firestore_store as store
from src.core.firebase_client import run_in_firestore_executor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Research Agent API",
    description="Autonomous research and architecture generation system",
    version="2.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state for live sessions (logs, streaming status)
# Completed sessions are persisted to Firestore
# ---------------------------------------------------------------------------
active_sessions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Firebase Auth middleware (optional — set FIREBASE_AUTH_ENABLED=true to enforce)
# ---------------------------------------------------------------------------

_AUTH_ENABLED = os.getenv("FIREBASE_AUTH_ENABLED", "false").lower() == "true"


async def verify_firebase_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Verify Firebase ID token from the Authorization header.
    Returns the user's UID, or None if auth is disabled.

    NOTE: ``auth.verify_id_token`` is a blocking network call.
    It is dispatched to the thread-pool executor so it never blocks
    the async event loop.
    """
    if not _AUTH_ENABLED:
        return None

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "")
    try:
        from firebase_admin import auth  # type: ignore[import-untyped]

        decoded = await run_in_firestore_executor(auth.verify_id_token, token)
        return decoded["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = 3
    search_provider: str = "tavily"
    session_id: Optional[str] = None
    mode: Optional[Literal["quick", "deep"]] = "deep"


class ResearchResponse(BaseModel):
    session_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Async-safe Firestore helpers — delegates to the centralised executor in
# src.core.firebase_client.  Kept as a local alias for readability.
# ---------------------------------------------------------------------------

_run_in_executor = run_in_firestore_executor


# ---------------------------------------------------------------------------
# Background research task
# ---------------------------------------------------------------------------

async def run_research_task_async(
    session_id: str,
    query: str,
    max_iterations: int,
    search_provider: str,
    mode: str,
    user_id: Optional[str] = None,
):
    """Run research asynchronously and persist results to Firestore."""
    try:
        # 1. Update in-memory status
        active_sessions[session_id]["status"] = "running"
        print(f"  Running session {session_id} in mode={mode}")

        # 2. Initialize Orchestrator
        orchestrator = DeepResearchOrchestratorV2(
            search_provider=search_provider,
            max_iterations=max_iterations,
            verbose=True,
            use_memory=True,
        )

        # 3. Load previous context from Firestore if session already exists
        previous_context = None
        try:
            existing_result = await _run_in_executor(store.get_results, session_id)
            existing_session = await _run_in_executor(store.get_session, session_id)
            if existing_result:
                previous_context = {
                    "messages": (existing_session or {}).get("messages", []),
                    "summary": existing_result.get("report"),
                    "evidence_graph": existing_result.get("evidence_summary", {}).get("evidence_graph"),
                    "task_graph": existing_result.get("task_graph_summary"),
                }
        except Exception as e:
            print(f"Context load failed for session {session_id}: {e}")

        # 4. Capture logs
        def distinct_log(message: str):
            print(f"[Task {session_id}] {message}")
            if session_id in active_sessions:
                active_sessions[session_id]["logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                })
        orchestrator.log = distinct_log

        # 5. Run Research
        result = await orchestrator.research_async(
            query=query,
            session_id=session_id,
            previous_context=previous_context,
            preferences={"mode": mode, "user_id": user_id},
        )

        # 6. Build persistence payloads
        final_data = {
            "report": result.report,
            "sources": result.sources,
            "metadata": {
                **result.metadata,
                "execution_stats": result.execution_stats,
                "iterations": result.iterations,
                "evidence_graph": result.evidence_graph,
                "task_graph": result.task_graph,
            },
        }

        # 7. Update in-memory
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["result"] = final_data

        # 8. Persist to Firestore
        try:
            evidence_summary = {
                "sources": result.sources,
                "evidence_graph": result.evidence_graph,
            }
            task_graph_summary = result.task_graph

            await _run_in_executor(
                store.save_results,
                session_id,
                result.report,
                evidence_summary,
                task_graph_summary,
            )

            # Save metrics
            metrics = result.metadata.get("metrics", {})
            metrics_payload = {
                "latency_ms": int(metrics.get("latency", 0) * 1000),
                "prompt_tokens": metrics.get("prompt_tokens", 0),
                "completion_tokens": metrics.get("completion_tokens", 0),
                "total_cost": metrics.get("cost_estimate", 0.0),
                "model_used": str(metrics.get("models_used", {})),
                "mode": mode,
            }
            await _run_in_executor(store.save_metrics, session_id, metrics_payload)

            print(f"✅ Session {session_id} saved to Firestore")
        except Exception as e:
            print(f"⚠️ Firestore persistence failed for {session_id}: {e}")

    except Exception as e:
        print(f"❌ Error in task {session_id}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "failed"
            active_sessions[session_id]["error"] = str(e)

        try:
            await _run_in_executor(store.update_session_status, session_id, "failed")
        except Exception:
            pass


def run_research_task_wrapper(
    session_id: str,
    query: str,
    max_iterations: int,
    search_provider: str,
    mode: str,
    user_id: Optional[str] = None,
):
    """Sync wrapper for BackgroundTasks."""
    asyncio.run(
        run_research_task_async(session_id, query, max_iterations, search_provider, mode, user_id)
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(verify_firebase_token),
):
    """Start a new research task or continue an existing session."""
    allowed_modes = {"quick", "deep"}
    mode = request.mode if request.mode in allowed_modes else "deep"
    if request.mode not in allowed_modes:
        print(f"⚠️ Invalid mode '{request.mode}' received; defaulting to 'deep'")
    print(f"Selected mode for session: {mode}")

    now_ts = datetime.now().isoformat()
    message_entry = {"role": "user", "content": request.query, "timestamp": now_ts}

    # --- Session resolution: reuse existing or create new ---
    session_id: str
    if request.session_id:
        # Continuation — load existing session
        existing = await _run_in_executor(store.get_session, request.session_id)
        if existing:
            session_id = request.session_id
            # Append message and update timestamp
            await _run_in_executor(
                store.update_session_field,
                session_id,
                status="running",
            )
        else:
            # Session ID provided but not found — create with that specific ID
            session_id = request.session_id
            try:
                await _run_in_executor(
                    store.create_session_with_id, session_id, user_id, request.query, mode,
                )
            except Exception as e:
                print(f"Session creation with custom ID failed: {e}")
    else:
        # New session
        session_id = await _run_in_executor(store.create_session, user_id, request.query, mode)

    # 1. Create/extend in-memory state
    active_sessions[session_id] = {
        "status": "pending",
        "query": request.query,
        "created_at": now_ts,
        "logs": [],
        "messages": [message_entry],
        "mode": mode,
    }

    # 2. Start Background Task
    background_tasks.add_task(
        run_research_task_wrapper,
        session_id,
        request.query,
        request.max_iterations,
        request.search_provider,
        mode,
        user_id,
    )

    return ResearchResponse(
        session_id=session_id,
        status="started",
        message="Research task started in background",
    )


@app.get("/api/research/{session_id}")
async def get_research_status(session_id: str):
    """Get status and results (Hybrid: in-memory → Firestore)."""

    # 1. Check active in-memory sessions (live)
    if session_id in active_sessions:
        return active_sessions[session_id]

    # 2. Check Firestore (history)
    try:
        session = await _run_in_executor(store.get_session, session_id)
        if session:
            result_data = await _run_in_executor(store.get_results, session_id)
            return {
                "status": session.get("status", "unknown"),
                "query": session.get("query", ""),
                "created_at": session.get("created_at"),
                "logs": [],
                "result": {
                    "report": result_data.get("report", "") if result_data else None,
                    "sources": (result_data.get("evidence_summary", {}) or {}).get("sources", []) if result_data else [],
                    "metadata": result_data.get("task_graph_summary", {}) if result_data else {},
                } if result_data else None,
            }
    except Exception as e:
        print(f"Firestore Fetch Error: {e}")

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/api/history")
async def get_history(user_id: Optional[str] = Depends(verify_firebase_token)):
    """List past research sessions from Firestore."""
    try:
        history = await _run_in_executor(store.get_research_history, user_id, 20)
        return history
    except Exception as e:
        print(f"History Fetch Error: {e}")
        return []


@app.get("/health")
async def health_check():
    """
    Health check endpoint for deployment monitoring.

    Probes **both** Firestore and Qdrant in parallel.  Returns a
    structured response so that load-balancers can act on individual
    component health.

    Status values:
        healthy   – all components reachable
        degraded  – at least one component unavailable
    """
    # --- Firestore probe (via centralized executor) ---
    async def _check_firestore() -> bool:
        try:
            from src.core.firebase_client import db as _db
            if _db is None:
                return False
            await run_in_firestore_executor(
                lambda: _db.collection("research_sessions").limit(1).get()
            )
            return True
        except Exception:
            return False

    # --- Qdrant probe (via centralized executor) ---
    async def _check_qdrant() -> bool:
        try:
            from src.memory.qdrant_store import QdrantClient as _QdrantClient
            import os as _os

            if _QdrantClient is None:
                return False
            url = _os.getenv("QDRANT_URL")
            api_key = _os.getenv("QDRANT_API_KEY")
            if url:
                client = _QdrantClient(url=url, api_key=api_key, timeout=5)
            else:
                client = _QdrantClient(host="localhost", port=6333, timeout=5)
            # get_collections is the lightest RPC that proves connectivity
            await run_in_firestore_executor(client.get_collections)
            return True
        except Exception:
            return False

    firestore_ok, qdrant_ok = await asyncio.gather(
        _check_firestore(), _check_qdrant()
    )

    overall = "healthy" if (firestore_ok and qdrant_ok) else "degraded"

    return {
        "status": overall,
        "service": "deep-research-agent",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "firestore": "ok" if firestore_ok else "unreachable",
            "qdrant": "ok" if qdrant_ok else "unreachable",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
