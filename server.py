from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import uuid
import asyncio
import os
from datetime import datetime

from main import DeepResearchOrchestratorV2, ResearchResultV2
from src.storage.supabase_store import SupabaseStorage
from src.memory.models import Session

app = FastAPI(title="Deep Research Agent API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active sessions (logs, streaming status)
# Completed sessions are persisted to Supabase
active_sessions: Dict[str, Dict[str, Any]] = {}

# Initialize Supabase Storage
try:
    storage = SupabaseStorage(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_KEY")
    )
    print("✅ Supabase Storage initialized")
except Exception as e:
    print(f"⚠️ Supabase Storage failed to initialize: {e}")
    storage = None

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

async def run_research_task_async(session_id: str, query: str, max_iterations: int, search_provider: str, mode: str):
    """Run research asynchronously and persist to DB."""
    try:
        # 1. Update in-memory status
        active_sessions[session_id]["status"] = "running"
        print(f" Running session {session_id} in mode={mode}")
        
        # 2. Update DB status (if available)
        if storage:
            # We assume session was created in pending state by the API endpoint
            pass 

        # 3. Initialize Orchestrator
        # Enable memory implementation to use Supabase storage internally
        orchestrator = DeepResearchOrchestratorV2(
            search_provider=search_provider,
            max_iterations=max_iterations,
            verbose=True,
            use_memory=True if storage else False
        )

        # 3b. Load previous context if session exists
        previous_context = None
        if storage:
            try:
                existing = await storage.get_session(session_id)
                if existing and existing.metadata:
                    meta = existing.metadata or {}
                    result_meta = meta.get("result", {})
                    previous_context = {
                        "messages": meta.get("messages", []),
                        "summary": result_meta.get("report"),
                        "evidence_graph": result_meta.get("metadata", {}).get("evidence_graph") if isinstance(result_meta, dict) else None,
                        "task_graph": result_meta.get("metadata", {}).get("task_graph") if isinstance(result_meta, dict) else None,
                    }
            except Exception as e:
                print(f"Context load failed for session {session_id}: {e}")
        
        # 4. Capture logs
        original_log = orchestrator.log
        def distinct_log(message: str):
            print(f"[Task {session_id}] {message}")
            if session_id in active_sessions:
                active_sessions[session_id]["logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "message": message
                })
        orchestrator.log = distinct_log
        
        # 5. Run Research
        # Pass session_id so orchestrator uses the same DB record
        result = await orchestrator.research_async(
            query=query, 
            session_id=session_id,
            previous_context=previous_context,
            preferences={"mode": mode},
        )
        
        # 6. Process Completion
        final_data = {
            "report": result.report,
            "sources": result.sources,
            "metadata": {
                **result.metadata,
                "execution_stats": result.execution_stats,
                "iterations": result.iterations,
                "evidence_graph": result.evidence_graph,
                "task_graph": result.task_graph
            }
        }
        
        # Update in-memory
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "completed"
            active_sessions[session_id]["result"] = final_data
            
        # Update DB persistence
        if storage:
            # Fetch current session to update it
            session = await storage.get_session(session_id)
            if session:
                session.status = "completed"
                # Store the full result in metadata so we can retrieve it later
                # In a fancier version, we rely on 'sources' table and 'summary_snapshots', 
                # but for the frontend to work easily, we dump the JSON result here.
                session.metadata["result"] = final_data
                await storage.update_session(session)
                print(f"✅ Session {session_id} saved to Supabase")

    except Exception as e:
        print(f"❌ Error in task {session_id}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "failed"
            active_sessions[session_id]["error"] = str(e)
            
        if storage:
            try:
                session = await storage.get_session(session_id)
                if session:
                    session.status = "failed"
                    session.metadata["error"] = str(e)
                    await storage.update_session(session)
            except:
                pass


def run_research_task_wrapper(session_id: str, query: str, max_iterations: int, search_provider: str, mode: str):
    """Sync wrapper for BackgroundTasks."""
    asyncio.run(run_research_task_async(session_id, query, max_iterations, search_provider, mode))


@app.post("/api/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new research task."""
    allowed_modes = {"quick", "deep"}
    mode = request.mode if request.mode in allowed_modes else "deep"
    if request.mode not in allowed_modes:
        print(f"⚠️ Invalid mode '{request.mode}' received; defaulting to 'deep'")
    print(f"Selected mode for session: {mode}")
    session_id = request.session_id or str(uuid.uuid4())
    now_ts = datetime.now().isoformat()
    message_entry = {"role": "user", "content": request.query, "timestamp": now_ts}
    
    # 1. Create/extend in-memory state
    active_sessions[session_id] = {
        "status": "pending",
        "query": request.query,
        "created_at": now_ts,
        "logs": [],
        "messages": [message_entry],
        "mode": mode,
    }
    
    # 2. Create or update DB state (Pending)
    if storage:
        try:
            existing = await storage.get_session(session_id)
            if existing:
                existing.status = "pending"
                existing.query_text = existing.query_text or request.query
                meta = existing.metadata or {}
                msgs = meta.get("messages", [])
                msgs.append(message_entry)
                meta["messages"] = msgs
                meta["mode"] = mode
                existing.metadata = meta
                await storage.update_session(existing)
            else:
                session = Session(
                    id=session_id,
                    query_text=request.query,
                    status="pending",
                    metadata={
                        "provider": request.search_provider,
                        "messages": [message_entry],
                        "mode": mode,
                    }
                )
                await storage.create_session(session)
        except Exception as e:
            print(f"Failed to create/update DB session: {e}")
    
    # 3. Start Background Task
    background_tasks.add_task(
        run_research_task_wrapper,
        session_id,
        request.query,
        request.max_iterations,
        request.search_provider,
        mode,
    )
    
    return ResearchResponse(
        session_id=session_id,
        status="started",
        message="Research task started in background"
    )

@app.get("/api/research/{session_id}")
async def get_research_status(session_id: str):
    """Get status and results (Hybrid: Memory -> DB)."""
    
    # 1. Check Active Memory (Live)
    if session_id in active_sessions:
        return active_sessions[session_id]
    
    # 2. Check Database (History)
    if storage:
        try:
            session = await storage.get_session(session_id)
            if session:
                return {
                    "status": session.status,
                    "query": session.query_text,
                    "created_at": session.created_at.isoformat(),
                    "logs": [], # DB logs not typically stored/fetched for history to save bandwidth
                    "result": session.metadata.get("result")
                }
        except Exception as e:
            print(f"DB Fetch Error: {e}")
            
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/history")
async def get_history():
    """List past research sessions."""
    if not storage:
        return []
        
    try:
        sessions = await storage.list_sessions(limit=20)
        return [
            {
                "id": s.id,
                "query": s.query_text,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "has_result": "result" in s.metadata
            }
            for s in sessions
        ]
    except Exception as e:
        print(f"History Fetch Error: {e}")
        return []

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "storage": "connected" if storage else "disabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
