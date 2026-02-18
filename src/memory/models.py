"""
Deep Research Agent V2 - Memory Data Models

Core data structures for the memory layer including:
- Claim: Factual statements extracted from sources
- Source: Information sources with metadata
- Session: Research session state
- SummarySnapshot: Compressed session snapshots
- EvidenceEdge: Relationships between claims and sources
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class EvidenceRelation(Enum):
    """Types of evidence relationships between claims and sources."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"


@dataclass
class Claim:
    """
    A factual claim extracted from source content.
    
    Claims are the atomic units of knowledge in the evidence graph.
    They are linked to sources via EvidenceEdges.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""                              # Canonical claim text
    normalized_text: str = ""                   # Deduplicated/canonicalized form
    embedding: Optional[List[float]] = None     # Vector embedding for similarity search
    provenance: List[str] = field(default_factory=list)  # source_ids that mention this
    confidence: float = 0.0                     # Aggregated confidence score
    supporting_text: str = ""                   # Original text that supports this claim
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "embedding": self.embedding,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "supporting_text": self.supporting_text,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            normalized_text=data.get("normalized_text", ""),
            embedding=data.get("embedding"),
            provenance=data.get("provenance", []),
            confidence=data.get("confidence", 0.0),
            supporting_text=data.get("supporting_text", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


@dataclass
class Source:
    """
    An information source (article, paper, webpage, etc.)
    
    Sources are validated and scored for reliability.
    They link to claims via the evidence graph.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    domain: str = ""
    title: str = ""
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    content_blob: str = ""                      # Path to stored raw content
    text_excerpt: str = ""                      # Cached first 2000 chars
    embedding: Optional[List[float]] = None     # Vector embedding
    reliability_score: float = 0.5              # 0-1 score from validation
    source_type: str = "web"                    # web, academic, technical
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "author": self.author,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "content_blob": self.content_blob,
            "text_excerpt": self.text_excerpt,
            "embedding": self.embedding,
            "reliability_score": self.reliability_score,
            "source_type": self.source_type,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            title=data.get("title", ""),
            author=data.get("author"),
            published_date=datetime.fromisoformat(data["published_date"]) if data.get("published_date") else None,
            content_blob=data.get("content_blob", ""),
            text_excerpt=data.get("text_excerpt", ""),
            embedding=data.get("embedding"),
            reliability_score=data.get("reliability_score", 0.5),
            source_type=data.get("source_type", "web"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class Session:
    """
    A research session tracking the state of an investigation.
    
    Sessions maintain context across multiple search iterations
    and link to snapshots for compression.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    query_text: str = ""
    summary_snapshot_id: Optional[str] = None
    iterations: int = 0
    status: str = "active"                      # active, complete, failed
    task_graph_id: Optional[str] = None         # Link to hierarchical planner task graph
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query_text": self.query_text,
            "summary_snapshot_id": self.summary_snapshot_id,
            "iterations": self.iterations,
            "status": self.status,
            "task_graph_id": self.task_graph_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            user_id=data.get("user_id"),
            query_text=data.get("query_text", ""),
            summary_snapshot_id=data.get("summary_snapshot_id"),
            iterations=data.get("iterations", 0),
            status=data.get("status", "active"),
            task_graph_id=data.get("task_graph_id"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


@dataclass
class SummarySnapshot:
    """
    A compressed snapshot of session state.
    
    Used for long-term memory compression to stay within
    LLM context limits while preserving key information.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    compressed_text: str = ""                   # Compressed summary (<=1024 tokens)
    embedding: Optional[List[float]] = None     # Vector for retrieval
    size_bytes: int = 0
    iteration_number: int = 0                   # Which iteration this snapshot represents
    claim_ids: List[str] = field(default_factory=list)  # Claims included in snapshot
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "compressed_text": self.compressed_text,
            "embedding": self.embedding,
            "size_bytes": self.size_bytes,
            "iteration_number": self.iteration_number,
            "claim_ids": self.claim_ids,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummarySnapshot":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            session_id=data.get("session_id", ""),
            compressed_text=data.get("compressed_text", ""),
            embedding=data.get("embedding"),
            size_bytes=data.get("size_bytes", 0),
            iteration_number=data.get("iteration_number", 0),
            claim_ids=data.get("claim_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


@dataclass
class EvidenceEdge:
    """
    A relationship between a claim and a source.
    
    Forms the edges of the evidence graph, linking
    claims to their supporting or contradicting sources.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_claim_id: str = ""
    to_source_id: str = ""
    relation: EvidenceRelation = EvidenceRelation.MENTIONS
    strength: float = 0.5                       # 0-1 strength of relationship
    validation_notes: str = ""                  # Notes from validation
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_claim_id": self.from_claim_id,
            "to_source_id": self.to_source_id,
            "relation": self.relation.value,
            "strength": self.strength,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceEdge":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            from_claim_id=data.get("from_claim_id", ""),
            to_source_id=data.get("to_source_id", ""),
            relation=EvidenceRelation(data.get("relation", "mentions")),
            strength=data.get("strength", 0.5),
            validation_notes=data.get("validation_notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )
