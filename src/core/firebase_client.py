"""
Deep Research Agent - Firebase Client

Initializes Firebase Admin SDK for Firestore and Auth.
Provides the shared Firestore database client used across the application.

If credentials are missing the module logs a warning and sets ``db = None``
so the rest of the app can fall back to in-memory storage instead of crashing
at import time.

Also exports ``run_in_firestore_executor`` — a centralised async helper that
dispatches any blocking (sync) callable to a shared ``ThreadPoolExecutor``.
All modules that call Firestore from async contexts should use this single
wrapper instead of creating their own executor.
"""

import asyncio
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared thread pool for ALL blocking Firestore / Firebase-Auth calls.
# Capped at 4 workers — enough for typical concurrency without exhausting
# Firestore connection limits.
# ---------------------------------------------------------------------------

_firestore_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("FIRESTORE_EXECUTOR_WORKERS", "4")),
    thread_name_prefix="firestore-io",
)

T = TypeVar("T")


async def run_in_firestore_executor(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run a **synchronous** callable on the shared Firestore thread pool.

    Usage::

        result = await run_in_firestore_executor(store.get_session, session_id)

    This is the single gateway for all blocking Firebase calls so that:
    * the async event loop is never blocked,
    * thread-pool size is controlled in one place,
    * callers don't need to manage their own executor.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _firestore_executor, lambda: fn(*args, **kwargs)
    )


# ---------------------------------------------------------------------------
# Firebase Admin Initialization (singleton-safe, soft-fail)
# ---------------------------------------------------------------------------

db = None  # will be set to a Firestore client on success

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth  # noqa: F401

    if not firebase_admin._apps:
        _cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase_key.json")
        if not os.path.exists(_cred_path):
            raise FileNotFoundError(
                f"Firebase credentials not found at '{_cred_path}'. "
                "Set FIREBASE_CREDENTIALS_PATH to the path of your service account JSON."
            )
        cred = credentials.Certificate(_cred_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized (credentials: %s)", _cred_path)

    # Shared Firestore client
    db = firestore.client()

except Exception as exc:
    logger.warning(
        "Firebase initialization failed — running WITHOUT Firestore persistence. "
        "Error: %s",
        exc,
    )
