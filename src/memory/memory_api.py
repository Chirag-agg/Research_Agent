"""
Deep Research Agent V2 - Unified Memory API

Provides a unified interface for all memory operations, coordinating:
- Supabase: Structured data (claims, sources, edges, sessions)
- Qdrant: Semantic memory (vector search)
"""

import asyncio
import inspect
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .models import Claim, Source, Session, SummarySnapshot, EvidenceEdge, EvidenceRelation
from .qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class MemoryAPI:
    """
    Unified Memory API for Deep Research Agent V2.
    
    Coordinates between:
    - Supabase storage for structured data persistence
    - Qdrant for semantic memory extraction and search
    
    Provides all memory operations needed by the research agents:
    - Session management
    - Source storage and deduplication
    - Claim extraction and linking
    - Evidence graph operations
    - Memory compression and retrieval
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        embedding_service: Optional[Any] = None,
    ):
        """
        Initialize the Memory API.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            embedding_service: Service for generating embeddings
        """
        # Try to import and initialize Supabase storage
        self.supabase = None
        try:
            from ..storage.supabase_store import SupabaseStorage
            self.supabase = SupabaseStorage(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
            )
        except (ImportError, ValueError) as e:
            print(f"Supabase storage not available: {e}")
        
        # Initialize Qdrant vector store for semantic memory
        try:
            self.vector_store = QdrantStore()
            logger.info("Vector store initialized", extra={"component": "memory", "backend": "qdrant"})
        except Exception as e:
            self.vector_store = None
            logger.error(
                "Vector store initialization failed; semantic storage disabled",
                extra={"component": "memory", "backend": "qdrant", "error": str(e)},
            )
        
        self.embedding_service = embedding_service
        
        # In-memory fallback for development
        self._sessions: Dict[str, Session] = {}
        self._sources: Dict[str, Source] = {}
        self._claims: Dict[str, Claim] = {}
        self._edges: Dict[str, EvidenceEdge] = {}
        self._snapshots: Dict[str, SummarySnapshot] = {}

    def _run_coro_sync(self, coro):
        """Safely run an async coroutine from sync code without crashing if a loop exists."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Dict[str, Any] = {}

        def runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result["value"] = new_loop.run_until_complete(coro)
            new_loop.close()

        import threading

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        return result.get("value")
    
    # ===============================
    # Session Management
    # ===============================
    
    async def create_session(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """
        Create a new research session.
        
        Args:
            query: The research query
            user_id: Optional user identifier
            session_id: Optional custom session ID
            
        Returns:
            Created Session object
        """
        session = Session(
            query_text=query,
            user_id=user_id,
        )
        if session_id:
            session.id = session_id
        
        if self.supabase:
            await self.supabase.create_session(session)
        else:
            self._sessions[session.id] = session
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        if self.supabase:
            return await self.supabase.get_session(session_id)
        return self._sessions.get(session_id)
    
    async def update_session(self, session: Session) -> None:
        """Update a session."""
        session.updated_at = datetime.now()
        if self.supabase:
            await self.supabase.update_session(session)
        else:
            self._sessions[session.id] = session
    
    async def increment_iteration(self, session_id: str) -> int:
        """Increment session iteration count."""
        session = await self.get_session(session_id)
        if session:
            session.iterations += 1
            await self.update_session(session)
            return session.iterations
        return 0
    
    # ===============================
    # Source Management
    # ===============================
    
    async def add_source(self, source: Source) -> str:
        """
        Add a source to storage.
        
        Automatically checks for duplicates by URL.
        Generates embedding if embedding service is available.
        
        Args:
            source: Source object to store
            
        Returns:
            Source ID
        """
        # Check for existing source by URL
        existing = await self.find_source_by_url(source.url)
        if existing:
            return existing.id
        
        # Generate embedding if service available
        if self.embedding_service and source.text_excerpt:
            source.embedding = await self.embedding_service.embed(source.text_excerpt)
        
        if self.supabase:
            await self.supabase.save_source(source)
        else:
            self._sources[source.id] = source
        
        return source.id
    
    async def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        if self.supabase:
            return await self.supabase.get_source(source_id)
        return self._sources.get(source_id)
    
    async def find_source_by_url(self, url: str) -> Optional[Source]:
        """Find a source by URL."""
        if self.supabase:
            return await self.supabase.search_sources_by_url(url)
        
        for source in self._sources.values():
            if source.url == url:
                return source
        return None
    
    async def get_sources_by_ids(self, source_ids: List[str]) -> List[Source]:
        """Get multiple sources by IDs."""
        if self.supabase:
            return await self.supabase.get_sources_by_ids(source_ids)
        return [self._sources[sid] for sid in source_ids if sid in self._sources]
    
    async def search_similar_sources(
        self,
        text: str,
        k: int = 10,
        threshold: float = 0.8,
    ) -> List[Tuple[Source, float]]:
        """
        Search for similar sources using semantic similarity.
        
        Args:
            text: Query text
            k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of (Source, similarity) tuples
        """
        if not self.embedding_service:
            return []
        
        embedding = await self.embedding_service.embed(text)
        
        if self.supabase:
            return await self.supabase.search_similar_sources(embedding, k, threshold)
        
        # Simple cosine similarity for in-memory fallback
        return []
    
    # ===============================
    # Claim Management
    # ===============================
    
    async def extract_claims(self, source_id: str, llm_client: Any = None) -> List[str]:
        """
        Extract claims from a source's content.
        
        This is typically called by the ClaimExtractorAgent.
        Returns claim IDs for the newly created claims.
        
        Args:
            source_id: ID of source to extract from
            llm_client: LLM client for extraction (optional)
            
        Returns:
            List of claim IDs
        """
        source = await self.get_source(source_id)
        if not source:
            return []
        
        # Claim extraction is handled by ClaimExtractorAgent
        # This method is a placeholder for the API
        return []
    
    async def add_claim(
        self,
        claim: Claim,
        source_id: str,
        relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
        strength: float = 0.5,
    ) -> str:
        """
        Add a claim and link it to a source.
        
        Performs deduplication by checking for similar existing claims.
        
        Args:
            claim: Claim object
            source_id: Source this claim comes from
            relation: Relationship type
            strength: Relationship strength
            
        Returns:
            Claim ID (may be existing if duplicate found)
        """
        # Generate embedding for deduplication
        if self.embedding_service and claim.text:
            claim.embedding = await self.embedding_service.embed(claim.text)
            
            # Check for duplicate claims
            duplicate = await self.deduplicate_claim(claim)
            if duplicate:
                # Update existing claim's provenance
                existing = await self.get_claim(duplicate)
                if existing and source_id not in existing.provenance:
                    existing.provenance.append(source_id)
                    await self.update_claim(existing)
                
                # Still create the evidence edge
                edge = EvidenceEdge(
                    from_claim_id=duplicate,
                    to_source_id=source_id,
                    relation=relation,
                    strength=strength,
                )
                await self.add_edge(edge)
                
                return duplicate
        
        # Add source to provenance
        claim.provenance.append(source_id)
        
        # Store claim
        if self.supabase:
            await self.supabase.save_claim(claim)
        else:
            self._claims[claim.id] = claim
        
        # Create evidence edge
        edge = EvidenceEdge(
            from_claim_id=claim.id,
            to_source_id=source_id,
            relation=relation,
            strength=strength,
        )
        await self.add_edge(edge)
        
        return claim.id
    
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        if self.supabase:
            return await self.supabase.get_claim(claim_id)
        return self._claims.get(claim_id)
    
    async def update_claim(self, claim: Claim) -> None:
        """Update a claim."""
        claim.updated_at = datetime.now()
        if self.supabase:
            await self.supabase.save_claim(claim)
        else:
            self._claims[claim.id] = claim
    
    async def deduplicate_claim(
        self,
        claim: Claim,
        threshold: float = 0.85,
    ) -> Optional[str]:
        """
        Check if a similar claim already exists.
        
        Args:
            claim: Claim to check
            threshold: Similarity threshold for deduplication
            
        Returns:
            Existing claim ID if duplicate found, None otherwise
        """
        if not claim.embedding:
            return None
        
        if self.supabase:
            similar = await self.supabase.search_similar_claims(
                claim.embedding,
                k=1,
                threshold=threshold,
            )
            if similar:
                return similar[0][0].id
        
        return None
    
    async def query_claims(
        self,
        text: str,
        k: int = 10,
        threshold: float = 0.7,
    ) -> List[Claim]:
        """
        Query claims by semantic similarity.
        
        Args:
            text: Query text
            k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of matching claims
        """
        if not self.embedding_service:
            return []
        
        embedding = await self.embedding_service.embed(text)
        
        if self.supabase:
            results = await self.supabase.search_similar_claims(embedding, k, threshold)
            return [claim for claim, _ in results]
        
        return []
    
    # ===============================
    # Evidence Graph Operations
    # ===============================
    
    async def add_edge(self, edge: EvidenceEdge) -> str:
        """Add an evidence edge to the graph."""
        if self.supabase:
            await self.supabase.save_edge(edge)
        else:
            self._edges[edge.id] = edge
        
        return edge.id
    
    async def get_evidence(self, claim_id: str) -> List[EvidenceEdge]:
        """Get all evidence edges for a claim."""
        if self.supabase:
            return await self.supabase.get_edges_by_claim(claim_id)
        return [e for e in self._edges.values() if e.from_claim_id == claim_id]
    
    async def get_supporting_sources(
        self,
        claim_id: str,
        n: int = 5,
    ) -> List[Source]:
        """
        Get sources that support a claim.
        
        Args:
            claim_id: Claim to get support for
            n: Maximum sources to return
            
        Returns:
            List of supporting sources, ordered by strength
        """
        if self.supabase:
            edges = await self.supabase.get_supporting_edges(claim_id)
        else:
            edges = [
                e for e in self._edges.values()
                if e.from_claim_id == claim_id and e.relation == EvidenceRelation.SUPPORTS
            ]
        
        # Sort by strength and get source IDs
        edges = sorted(edges, key=lambda e: e.strength, reverse=True)[:n]
        source_ids = [e.to_source_id for e in edges]
        
        return await self.get_sources_by_ids(source_ids)
    
    async def get_contradicting_sources(
        self,
        claim_id: str,
        n: int = 5,
    ) -> List[Source]:
        """Get sources that contradict a claim."""
        if self.supabase:
            edges = await self.supabase.get_contradicting_edges(claim_id)
        else:
            edges = [
                e for e in self._edges.values()
                if e.from_claim_id == claim_id and e.relation == EvidenceRelation.CONTRADICTS
            ]
        
        edges = sorted(edges, key=lambda e: e.strength, reverse=True)[:n]
        source_ids = [e.to_source_id for e in edges]
        
        return await self.get_sources_by_ids(source_ids)
    
    async def calculate_claim_confidence(self, claim_id: str) -> float:
        """
        Calculate aggregated confidence score for a claim.
        
        Based on:
        - Number and strength of supporting edges
        - Number and strength of contradicting edges
        - Reliability scores of linked sources
        
        Returns:
            Confidence score between 0 and 1
        """
        edges = await self.get_evidence(claim_id)
        if not edges:
            return 0.0
        
        support_score = 0.0
        contradict_score = 0.0
        
        for edge in edges:
            source = await self.get_source(edge.to_source_id)
            reliability = source.reliability_score if source else 0.5
            
            if edge.relation == EvidenceRelation.SUPPORTS:
                support_score += edge.strength * reliability
            elif edge.relation == EvidenceRelation.CONTRADICTS:
                contradict_score += edge.strength * reliability
        
        total = support_score + contradict_score
        if total == 0:
            return 0.0
        
        confidence = support_score / total
        
        # Update claim's confidence
        claim = await self.get_claim(claim_id)
        if claim:
            claim.confidence = confidence
            await self.update_claim(claim)
        
        return confidence
    
    # ===============================
    # Snapshot Operations
    # ===============================
    
    async def create_snapshot(
        self,
        session_id: str,
        compressed_text: str,
        claim_ids: List[str],
    ) -> str:
        """
        Create a compressed snapshot of session state.
        
        Args:
            session_id: Session to snapshot
            compressed_text: Compressed summary text
            claim_ids: Claims included in this snapshot
            
        Returns:
            Snapshot ID
        """
        session = await self.get_session(session_id)
        iteration = session.iterations if session else 0
        
        snapshot = SummarySnapshot(
            session_id=session_id,
            compressed_text=compressed_text,
            size_bytes=len(compressed_text.encode()),
            iteration_number=iteration,
            claim_ids=claim_ids,
        )
        
        # Generate embedding for snapshot retrieval
        if self.embedding_service and compressed_text:
            snapshot.embedding = await self.embedding_service.embed(compressed_text)
        
        if self.supabase:
            await self.supabase.save_snapshot(snapshot)
        else:
            self._snapshots[snapshot.id] = snapshot
        
        # Update session with latest snapshot
        if session:
            session.summary_snapshot_id = snapshot.id
            await self.update_session(session)
        
        return snapshot.id
    
    async def get_snapshots(self, session_id: str) -> List[SummarySnapshot]:
        """Get all snapshots for a session."""
        if self.supabase:
            return await self.supabase.get_snapshots(session_id)
        return [s for s in self._snapshots.values() if s.session_id == session_id]
    
    # ===============================
    # Semantic Memory (Qdrant)
    # ===============================
    
    def store_research_findings(
        self,
        session_id: str,
        query: str,
        findings: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store research findings as semantic memories via Qdrant.
        
        Args:
            session_id: Research session ID
            query: Original query
            findings: Validated findings
            user_id: User identifier
            
        Returns:
            List of memory IDs
        """
        if not self.vector_store or not self.embedding_service:
            logger.warning(
                "Skipping semantic store (vector_store or embedding missing)",
                extra={"component": "memory", "backend": "qdrant", "action": "store_findings"},
            )
            return []

        memory_ids: List[str] = []
        for i, finding in enumerate(findings):
            text = finding.get("content") or finding.get("text") or finding.get("snippet") or ""
            if not text:
                continue
            vector = None
            embed_fn = getattr(self.embedding_service, "embed", None)
            embed_sync_fn = getattr(self.embedding_service, "embed_sync", None)
            try:
                if embed_sync_fn and callable(embed_sync_fn):
                    vector = embed_sync_fn(text)
                elif embed_fn and callable(embed_fn):
                    if inspect.iscoroutinefunction(embed_fn):
                        logger.warning("Embedding service provides async embed; skipping store in sync context")
                    else:
                        vector = embed_fn(text)
            except Exception as e:
                logger.warning("Embedding failed for finding %s: %s", i, e)
                continue

            if vector is None:
                continue

            memory_id = finding.get("id") or f"{session_id}-{i}"
            payload = {
                "session_id": session_id,
                "query": query,
                "user_id": user_id,
                "finding": finding,
            }
            try:
                if asyncio.iscoroutinefunction(self.vector_store.upsert):
                    self._run_coro_sync(self.vector_store.upsert(memory_id, vector, payload))
                else:
                    self.vector_store.upsert(memory_id, vector, payload)  # type: ignore
                memory_ids.append(memory_id)
            except Exception as e:
                logger.warning("Vector store upsert failed for %s: %s", memory_id, e)
                continue

        return memory_ids
    
    async def recall_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories for a query via vector search.
        
        Args:
            query: Search query
            user_id: User context
            top_k: Number of results
            
        Returns:
            List of relevant memories (dicts with id, score, payload)
        """
        if not self.vector_store or not self.embedding_service:
            logger.warning(
                "Skipping recall (vector_store or embedding missing)",
                extra={"component": "memory", "backend": "qdrant", "action": "recall"},
            )
            return []

        try:
            vector = await self.embedding_service.embed(query)
        except Exception as e:
            logger.warning("Embedding failed during recall: %s", e)
            return []

        try:
            results = await self.vector_store.search(vector, top_k=top_k)
            return results
        except Exception as e:
            logger.warning("Vector store search failed: %s", e)
            return []
    
    async def get_session_memories(self, session_id: str) -> List[Dict[str, Any]]:
        """Get memories for a session (vector search fallback)."""
        if not self.vector_store:
            logger.warning(
                "Skipping session memories (vector_store missing)",
                extra={"component": "memory", "backend": "qdrant", "action": "session_memories"},
            )
            return []
        # Simple approach: search broadly and filter by session_id
        try:
            dummy_vector = [0.0] * getattr(self.vector_store, "embedding_dim", 1536)
            hits = await self.vector_store.search(dummy_vector, top_k=50)
            return [h for h in hits if h.get("payload", {}).get("session_id") == session_id]
        except Exception as e:
            logger.warning("Vector store session retrieval failed: %s", e)
            return []
