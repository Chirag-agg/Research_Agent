"""
Deep Research Agent - Firestore Storage Layer

Provides persistent storage for sessions, results, metrics, and user preferences
using Google Cloud Firestore. Replaces the former Supabase storage layer.

Collections:
    research_sessions/{session_id}   – session metadata & status
    research_results/{session_id}    – final report & evidence summary
    research_metrics/{session_id}    – latency, tokens, cost, model info
    users/{user_id}                  – user preferences

Design constraints:
    • Firestore document limit is 1 MB.  Reports exceeding 800 000 characters
      are truncated in Firestore; the full version is written to disk and a
      reference path is stored instead.
    • All public methods are synchronous but call Firestore through a thread
      executor so they are safe to invoke from async FastAPI handlers.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.cloud.firestore_v1 import FieldFilter  # type: ignore[import-untyped]
from google.cloud import firestore as firestore_module  # type: ignore[import-untyped]

from ..core.firebase_client import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIRESTORE_CHAR_LIMIT = 800_000  # conservative limit for 1 MB docs
_OVERFLOW_DIR = os.getenv("REPORT_OVERFLOW_DIR", "./report_overflow")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_db():
    """Raise early if Firestore is not available."""
    if db is None:
        raise RuntimeError(
            "Firestore client is not initialised. "
            "Check FIREBASE_CREDENTIALS_PATH and firebase_client.py logs."
        )


def _now() -> datetime:
    return datetime.utcnow()


def _ensure_overflow_dir() -> str:
    os.makedirs(_OVERFLOW_DIR, exist_ok=True)
    return _OVERFLOW_DIR


def _overflow_to_disk(session_id: str, label: str, data: str) -> str:
    """Write *data* to an overflow file and return the file path."""
    overflow_dir = _ensure_overflow_dir()
    overflow_path = os.path.join(overflow_dir, f"{session_id}_{label}.json")
    with open(overflow_path, "w", encoding="utf-8") as fh:
        fh.write(data)
    logger.warning(
        "%s for session %s overflows Firestore limit (%d chars). "
        "Saved to %s.",
        label, session_id, len(data), overflow_path,
    )
    return overflow_path


def _truncate_report(session_id: str, report: str) -> Dict[str, Any]:
    """
    If *report* exceeds the Firestore character limit, write the full version
    to disk and return a dict with a truncated copy + overflow path.
    Otherwise return the report unchanged.
    """
    if len(report) <= _FIRESTORE_CHAR_LIMIT:
        return {"report": report, "report_overflow_path": None}

    overflow_path = _overflow_to_disk(session_id, "report", report)
    return {
        "report": report[:_FIRESTORE_CHAR_LIMIT] + "\n\n[TRUNCATED — full report stored externally]",
        "report_overflow_path": overflow_path,
    }


def _safe_payload_size(data: Any) -> int:
    """Estimate serialized size of *data* in characters (conservative)."""
    try:
        return len(json.dumps(data, default=str))
    except (TypeError, ValueError):
        return sys.getsizeof(data)


def _guard_evidence(
    session_id: str,
    evidence_summary: Dict[str, Any],
    task_graph_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Ensure evidence_summary and task_graph_summary fit inside a single
    Firestore document.  If either is too large, overflow to disk and
    replace the value with a reference path.

    Returns a dict with the (possibly trimmed) payloads and any overflow paths.
    """
    result: Dict[str, Any] = {
        "evidence_summary": evidence_summary,
        "task_graph_summary": task_graph_summary,
        "evidence_overflow_path": None,
        "task_graph_overflow_path": None,
    }

    # Budget: allocate 60% to evidence, 20% to task_graph, 20% to report + metadata
    evidence_budget = int(_FIRESTORE_CHAR_LIMIT * 0.6)
    task_graph_budget = int(_FIRESTORE_CHAR_LIMIT * 0.2)

    if _safe_payload_size(evidence_summary) > evidence_budget:
        path = _overflow_to_disk(
            session_id, "evidence", json.dumps(evidence_summary, default=str)
        )
        result["evidence_summary"] = {"_overflowed": True, "overflow_path": path}
        result["evidence_overflow_path"] = path

    if _safe_payload_size(task_graph_summary) > task_graph_budget:
        path = _overflow_to_disk(
            session_id, "task_graph", json.dumps(task_graph_summary, default=str)
        )
        result["task_graph_summary"] = {"_overflowed": True, "overflow_path": path}
        result["task_graph_overflow_path"] = path

    return result


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------

def _session_payload(user_id: Optional[str], query: str, mode: str, chat_name: Optional[str] = None) -> Dict[str, Any]:
    """Common document payload for new sessions."""
    payload = {
        "user_id": user_id or "",
        "query": query,
        "mode": mode,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "running",
        "has_result": False,   # flipped to True inside save_results transaction
    }
    if chat_name:
        payload["chat_name"] = chat_name
    return payload


def create_session(user_id: Optional[str], query: str, mode: str, chat_name: Optional[str] = None) -> str:
    """
    Create a new research session document.

    Returns:
        The generated session_id (UUID).
    """
    _require_db()
    session_id = str(uuid.uuid4())
    try:
        db.collection("research_sessions").document(session_id).set(
            _session_payload(user_id, query, mode, chat_name)
        )
        logger.info("Firestore session created: %s", session_id)
    except Exception as exc:
        logger.error("Firestore create_session failed: %s", exc)
        raise
    return session_id


def create_session_with_id(
    session_id: str, user_id: Optional[str], query: str, mode: str, chat_name: Optional[str] = None,
) -> str:
    """
    Create a session with a **caller-supplied** ID (e.g. for session
    continuation when the original doc is missing).

    Returns:
        The same *session_id* that was passed in.
    """
    _require_db()
    try:
        db.collection("research_sessions").document(session_id).set(
            _session_payload(user_id, query, mode, chat_name)
        )
        logger.info("Firestore session created (custom ID): %s", session_id)
    except Exception as exc:
        logger.error("Firestore create_session_with_id failed: %s", exc)
        raise
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session document.  Returns ``None`` if not found."""
    _require_db()
    try:
        doc = db.collection("research_sessions").document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as exc:
        logger.error("Firestore get_session failed: %s", exc)
        return None


def update_session_status(session_id: str, status: str) -> None:
    """
    Set session status (running | complete | failed).

    Uses a Firestore **transaction** so concurrent updates never
    silently overwrite each other.
    """
    _require_db()
    try:
        @firestore_module.transactional
        def _txn(transaction):
            ref = db.collection("research_sessions").document(session_id)
            transaction.update(ref, {
                "status": status,
                "updated_at": _now(),
            })

        _txn(db.transaction())
    except Exception as exc:
        logger.error("Firestore update_session_status failed: %s", exc)


def update_session_field(session_id: str, **fields: Any) -> None:
    """
    Update arbitrary fields on a session document.

    Uses a Firestore **transaction** to guarantee atomicity.
    """
    _require_db()
    try:
        fields["updated_at"] = _now()

        @firestore_module.transactional
        def _txn(transaction):
            ref = db.collection("research_sessions").document(session_id)
            transaction.update(ref, fields)

        _txn(db.transaction())
    except Exception as exc:
        logger.error("Firestore update_session_field failed: %s", exc)


def list_sessions(user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """List recent sessions, optionally filtered by user."""
    _require_db()
    try:
        ref = db.collection("research_sessions")
        if user_id:
            ref = ref.where(filter=FieldFilter("user_id", "==", user_id))
        docs = ref.order_by("created_at", direction="DESCENDING").limit(limit).stream()
        return [{**doc.to_dict(), "id": doc.id} for doc in docs]
    except Exception as exc:
        logger.error("Firestore list_sessions failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Results operations
# ---------------------------------------------------------------------------

def save_results(
    session_id: str,
    report: str,
    evidence_summary: Optional[Dict[str, Any]] = None,
    task_graph_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist the final research output.

    • Large reports are truncated; full version overflows to disk.
    • Large evidence / task_graph payloads are also overflowed to prevent
      hitting the Firestore 1 MB document limit.
    • The result write and session status update are executed inside a
      Firestore **transaction** to prevent partial writes on contention.
    """
    _require_db()
    try:
        report_data = _truncate_report(session_id, report)
        guarded = _guard_evidence(
            session_id,
            evidence_summary or {},
            task_graph_summary or {},
        )

        payload: Dict[str, Any] = {
            "report": report_data["report"],
            "evidence_summary": guarded["evidence_summary"],
            "task_graph_summary": guarded["task_graph_summary"],
        }
        if report_data["report_overflow_path"]:
            payload["report_overflow_path"] = report_data["report_overflow_path"]
        if guarded["evidence_overflow_path"]:
            payload["evidence_overflow_path"] = guarded["evidence_overflow_path"]
        if guarded["task_graph_overflow_path"]:
            payload["task_graph_overflow_path"] = guarded["task_graph_overflow_path"]

        # --- Atomic transaction: write results + mark session complete ---
        @firestore_module.transactional
        def _commit_results(transaction):
            result_ref = db.collection("research_results").document(session_id)
            session_ref = db.collection("research_sessions").document(session_id)
            transaction.set(result_ref, payload)
            transaction.update(session_ref, {
                "status": "complete",
                "has_result": True,
                "updated_at": _now(),
            })

        _commit_results(db.transaction())
        logger.info("Firestore results saved (transactional) for session %s", session_id)
    except Exception as exc:
        logger.error("Firestore save_results failed for %s: %s", session_id, exc)
        update_session_status(session_id, "failed")
        raise


def get_results(session_id: str) -> Optional[Dict[str, Any]]:
    """Load persisted results for a session."""
    _require_db()
    try:
        doc = db.collection("research_results").document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as exc:
        logger.error("Firestore get_results failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Metrics operations
# ---------------------------------------------------------------------------

def save_metrics(session_id: str, metrics: Dict[str, Any]) -> None:
    """Save execution metrics for a research session."""
    _require_db()
    try:
        db.collection("research_metrics").document(session_id).set(metrics)
    except Exception as exc:
        logger.error("Firestore save_metrics failed for %s: %s", session_id, exc)


def get_metrics(session_id: str) -> Optional[Dict[str, Any]]:
    """Load metrics for a session."""
    _require_db()
    try:
        doc = db.collection("research_metrics").document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as exc:
        logger.error("Firestore get_metrics failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# User preferences
# ---------------------------------------------------------------------------

def save_user_preferences(user_id: str, preferences: Dict[str, Any]) -> None:
    """Create or update user preferences."""
    _require_db()
    try:
        db.collection("users").document(user_id).set({
            "preferences": preferences,
            "updated_at": _now(),
        }, merge=True)
    except Exception as exc:
        logger.error("Firestore save_user_preferences failed for %s: %s", user_id, exc)


def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """Load user preferences.  Returns empty dict if not found."""
    _require_db()
    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            return doc.to_dict().get("preferences", {})
    except Exception as exc:
        logger.error("Firestore get_user_preferences failed: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Research history (convenience)
# ---------------------------------------------------------------------------

def get_research_history(user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return a summary list of past sessions with result availability.

    Uses the ``has_result`` flag stored on each session document
    (set atomically inside ``save_results``) to avoid the previous
    O(N) per-session ``get_results`` round-trip.
    """
    sessions = list_sessions(user_id=user_id, limit=limit)
    return [
        {
            "id": s.get("id", ""),
            "query": s.get("query", ""),
            "mode": s.get("mode", "deep"),
            "status": s.get("status", "unknown"),
            "created_at": s.get("created_at"),
            "has_result": s.get("has_result", False),
            "chat_name": s.get("chat_name"),  # Include stored chat name if available
        }
        for s in sessions
    ]

# ---------------------------------------------------------------------------
# Deletion operations
# ---------------------------------------------------------------------------

def delete_session(session_id: str) -> bool:
    """
    Delete a research session and all associated data (results, metrics).
    
    Uses a Firestore **batch** to ensure all documents are deleted atomically.
    
    Returns:
        True if deletion was successful, False otherwise.
    """
    _require_db()
    try:
        batch = db.batch()
        
        # Delete session document
        session_ref = db.collection("research_sessions").document(session_id)
        batch.delete(session_ref)
        
        # Delete results document if it exists
        results_ref = db.collection("research_results").document(session_id)
        batch.delete(results_ref)
        
        # Delete metrics document if it exists
        metrics_ref = db.collection("research_metrics").document(session_id)
        batch.delete(metrics_ref)
        
        batch.commit()
        logger.info("Firestore session deleted: %s", session_id)
        return True
    except Exception as exc:
        logger.error("Firestore delete_session failed for %s: %s", session_id, exc)
        return False