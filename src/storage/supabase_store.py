"""
Deep Research Agent V2 - Supabase Storage Layer

Provides persistent storage for the memory layer using:
- Supabase Postgres with pgvector extension for embeddings
- Supabase Storage for blob content (raw documents)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

from ..memory.models import Claim, Source, Session, SummarySnapshot, EvidenceEdge, EvidenceRelation

logger = logging.getLogger(__name__)


class SupabaseStorage:
    """
    Supabase-backed storage for claims, sources, sessions, and evidence edges.
    
    Uses pgvector for embedding similarity search and standard tables
    for relational data.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        """
        Initialize Supabase client.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (anon or service role)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        if create_client is None:
            raise ImportError("supabase-py not installed. Run: pip install supabase")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
    
    # ===============================
    # Session Operations
    # ===============================
    
    async def create_session(self, session: Session) -> str:
        """Create a new research session."""
        data = session.to_dict()
        try:
            self.client.table("sessions").upsert(data).execute()
        except Exception as e:
            logger.warning("Session persistence failed, continuing without persistence. Error: %s", e)
        return session.id
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        result = self.client.table("sessions").select("*").eq("id", session_id).execute()
        if result.data:
            return Session.from_dict(result.data[0])
        return None
    
    async def update_session(self, session: Session) -> None:
        """Update an existing session."""
        session.updated_at = datetime.now()
        data = session.to_dict()
        try:
            self.client.table("sessions").update(data).eq("id", session.id).execute()
        except Exception as e:
            logger.warning("Session persistence failed, continuing without persistence. Error: %s", e)
    
    async def list_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Session]:
        """List sessions, optionally filtered by user."""
        query = self.client.table("sessions").select("*").order("created_at", desc=True).limit(limit)
        if user_id:
            query = query.eq("user_id", user_id)
        result = query.execute()
        return [Session.from_dict(row) for row in result.data]
    
    # ===============================
    # Source Operations
    # ===============================
    
    async def save_source(self, source: Source) -> str:
        """Save a source to the database."""
        data = source.to_dict()
        # Handle embedding separately for pgvector
        embedding = data.pop("embedding", None)
        
        result = self.client.table("sources").upsert(data).execute()
        
        # Store embedding in vector table if present
        if embedding:
            await self._store_source_embedding(source.id, embedding)
        
        return source.id
    
    async def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        result = self.client.table("sources").select("*").eq("id", source_id).execute()
        if result.data:
            source_data = result.data[0]
            # Fetch embedding
            embedding = await self._get_source_embedding(source_id)
            source_data["embedding"] = embedding
            return Source.from_dict(source_data)
        return None
    
    async def get_sources_by_ids(self, source_ids: List[str]) -> List[Source]:
        """Get multiple sources by IDs."""
        if not source_ids:
            return []
        result = self.client.table("sources").select("*").in_("id", source_ids).execute()
        return [Source.from_dict(row) for row in result.data]
    
    async def search_sources_by_url(self, url: str) -> Optional[Source]:
        """Find a source by URL to avoid duplicates."""
        result = self.client.table("sources").select("*").eq("url", url).execute()
        if result.data:
            return Source.from_dict(result.data[0])
        return None
    
    async def search_similar_sources(
        self, 
        embedding: List[float], 
        k: int = 10,
        threshold: float = 0.8
    ) -> List[Tuple[Source, float]]:
        """
        Search for similar sources using vector similarity.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (Source, similarity_score) tuples
        """
        # Use Supabase RPC for pgvector similarity search
        result = self.client.rpc(
            "match_sources",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": k
            }
        ).execute()
        
        sources = []
        for row in result.data:
            source = await self.get_source(row["id"])
            if source:
                sources.append((source, row["similarity"]))
        
        return sources
    
    async def _store_source_embedding(self, source_id: str, embedding: List[float]) -> None:
        """Store source embedding in vector table."""
        self.client.table("source_embeddings").upsert({
            "source_id": source_id,
            "embedding": embedding
        }).execute()
    
    async def _get_source_embedding(self, source_id: str) -> Optional[List[float]]:
        """Get source embedding from vector table."""
        result = self.client.table("source_embeddings").select("embedding").eq("source_id", source_id).execute()
        if result.data:
            return result.data[0]["embedding"]
        return None
    
    # ===============================
    # Claim Operations
    # ===============================
    
    async def save_claim(self, claim: Claim) -> str:
        """Save a claim to the database."""
        data = claim.to_dict()
        embedding = data.pop("embedding", None)
        
        result = self.client.table("claims").upsert(data).execute()
        
        if embedding:
            await self._store_claim_embedding(claim.id, embedding)
        
        return claim.id
    
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        result = self.client.table("claims").select("*").eq("id", claim_id).execute()
        if result.data:
            claim_data = result.data[0]
            embedding = await self._get_claim_embedding(claim_id)
            claim_data["embedding"] = embedding
            return Claim.from_dict(claim_data)
        return None
    
    async def get_claims_by_session(self, session_id: str) -> List[Claim]:
        """Get all claims associated with a session."""
        # Get via snapshot -> claim_ids
        result = self.client.table("summary_snapshots").select("claim_ids").eq("session_id", session_id).execute()
        claim_ids = []
        for row in result.data:
            claim_ids.extend(row.get("claim_ids", []))
        
        if not claim_ids:
            return []
        
        claims_result = self.client.table("claims").select("*").in_("id", list(set(claim_ids))).execute()
        return [Claim.from_dict(row) for row in claims_result.data]
    
    async def search_similar_claims(
        self,
        embedding: List[float],
        k: int = 10,
        threshold: float = 0.85
    ) -> List[Tuple[Claim, float]]:
        """
        Search for similar claims using vector similarity.
        Used for deduplication (threshold=0.85) and retrieval.
        """
        result = self.client.rpc(
            "match_claims",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": k
            }
        ).execute()
        
        claims = []
        for row in result.data:
            claim = await self.get_claim(row["id"])
            if claim:
                claims.append((claim, row["similarity"]))
        
        return claims
    
    async def update_claim_confidence(self, claim_id: str, confidence: float) -> None:
        """Update the aggregated confidence score for a claim."""
        self.client.table("claims").update({
            "confidence": confidence,
            "updated_at": datetime.now().isoformat()
        }).eq("id", claim_id).execute()
    
    async def _store_claim_embedding(self, claim_id: str, embedding: List[float]) -> None:
        """Store claim embedding in vector table."""
        self.client.table("claim_embeddings").upsert({
            "claim_id": claim_id,
            "embedding": embedding
        }).execute()
    
    async def _get_claim_embedding(self, claim_id: str) -> Optional[List[float]]:
        """Get claim embedding from vector table."""
        result = self.client.table("claim_embeddings").select("embedding").eq("claim_id", claim_id).execute()
        if result.data:
            return result.data[0]["embedding"]
        return None
    
    # ===============================
    # Evidence Edge Operations
    # ===============================
    
    async def save_edge(self, edge: EvidenceEdge) -> str:
        """Save an evidence edge to the database."""
        data = edge.to_dict()
        self.client.table("evidence_edges").upsert(data).execute()
        return edge.id
    
    async def get_edges_by_claim(self, claim_id: str) -> List[EvidenceEdge]:
        """Get all evidence edges for a claim."""
        result = self.client.table("evidence_edges").select("*").eq("from_claim_id", claim_id).execute()
        return [EvidenceEdge.from_dict(row) for row in result.data]
    
    async def get_edges_by_source(self, source_id: str) -> List[EvidenceEdge]:
        """Get all evidence edges pointing to a source."""
        result = self.client.table("evidence_edges").select("*").eq("to_source_id", source_id).execute()
        return [EvidenceEdge.from_dict(row) for row in result.data]
    
    async def get_supporting_edges(self, claim_id: str) -> List[EvidenceEdge]:
        """Get edges where sources support the claim."""
        result = (
            self.client.table("evidence_edges")
            .select("*")
            .eq("from_claim_id", claim_id)
            .eq("relation", EvidenceRelation.SUPPORTS.value)
            .execute()
        )
        return [EvidenceEdge.from_dict(row) for row in result.data]
    
    async def get_contradicting_edges(self, claim_id: str) -> List[EvidenceEdge]:
        """Get edges where sources contradict the claim."""
        result = (
            self.client.table("evidence_edges")
            .select("*")
            .eq("from_claim_id", claim_id)
            .eq("relation", EvidenceRelation.CONTRADICTS.value)
            .execute()
        )
        return [EvidenceEdge.from_dict(row) for row in result.data]
    
    # ===============================
    # Snapshot Operations
    # ===============================
    
    async def save_snapshot(self, snapshot: SummarySnapshot) -> str:
        """Save a summary snapshot."""
        data = snapshot.to_dict()
        embedding = data.pop("embedding", None)
        
        self.client.table("summary_snapshots").upsert(data).execute()
        
        if embedding:
            await self._store_snapshot_embedding(snapshot.id, embedding)
        
        return snapshot.id
    
    async def get_snapshots(self, session_id: str) -> List[SummarySnapshot]:
        """Get all snapshots for a session."""
        result = (
            self.client.table("summary_snapshots")
            .select("*")
            .eq("session_id", session_id)
            .order("iteration_number")
            .execute()
        )
        return [SummarySnapshot.from_dict(row) for row in result.data]
    
    async def _store_snapshot_embedding(self, snapshot_id: str, embedding: List[float]) -> None:
        """Store snapshot embedding."""
        self.client.table("snapshot_embeddings").upsert({
            "snapshot_id": snapshot_id,
            "embedding": embedding
        }).execute()


# SQL to create tables (run in Supabase SQL editor)
SETUP_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    query_text TEXT NOT NULL,
    summary_snapshot_id UUID,
    iterations INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    task_graph_id UUID,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sources table
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    domain TEXT,
    title TEXT,
    author TEXT,
    published_date TIMESTAMPTZ,
    content_blob TEXT,
    text_excerpt TEXT,
    reliability_score FLOAT DEFAULT 0.5,
    source_type TEXT DEFAULT 'web',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(url)
);

-- Source embeddings with pgvector
CREATE TABLE IF NOT EXISTS source_embeddings (
    source_id UUID PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    embedding vector(1536)  -- OpenAI embedding dimension
);

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    normalized_text TEXT,
    provenance TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 0.0,
    supporting_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Claim embeddings with pgvector
CREATE TABLE IF NOT EXISTS claim_embeddings (
    claim_id UUID PRIMARY KEY REFERENCES claims(id) ON DELETE CASCADE,
    embedding vector(1536)
);

-- Evidence edges (linking claims to sources)
CREATE TABLE IF NOT EXISTS evidence_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    to_source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    relation TEXT NOT NULL CHECK (relation IN ('supports', 'contradicts', 'mentions')),
    strength FLOAT DEFAULT 0.5,
    validation_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(from_claim_id, to_source_id, relation)
);

-- Summary snapshots
CREATE TABLE IF NOT EXISTS summary_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    compressed_text TEXT,
    size_bytes INTEGER DEFAULT 0,
    iteration_number INTEGER DEFAULT 0,
    claim_ids TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Snapshot embeddings
CREATE TABLE IF NOT EXISTS snapshot_embeddings (
    snapshot_id UUID PRIMARY KEY REFERENCES summary_snapshots(id) ON DELETE CASCADE,
    embedding vector(1536)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);
CREATE INDEX IF NOT EXISTS idx_edges_claim ON evidence_edges(from_claim_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON evidence_edges(to_source_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON summary_snapshots(session_id);

-- Vector similarity search functions
CREATE OR REPLACE FUNCTION match_sources(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (id UUID, similarity float)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        se.source_id as id,
        1 - (se.embedding <=> query_embedding) as similarity
    FROM source_embeddings se
    WHERE 1 - (se.embedding <=> query_embedding) > match_threshold
    ORDER BY se.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION match_claims(
    query_embedding vector(1536),
    match_threshold float,
    match_count int
)
RETURNS TABLE (id UUID, similarity float)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ce.claim_id as id,
        1 - (ce.embedding <=> query_embedding) as similarity
    FROM claim_embeddings ce
    WHERE 1 - (ce.embedding <=> query_embedding) > match_threshold
    ORDER BY ce.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


def get_setup_sql() -> str:
    """Return the SQL needed to set up the database schema."""
    return SETUP_SQL
