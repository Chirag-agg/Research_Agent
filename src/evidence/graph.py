"""
Deep Research Agent V2 - Evidence Graph

Manages the claim-source evidence graph for transparent reasoning.
Provides graph operations for tracing claim provenance and calculating confidence.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..memory.models import Claim, Source, EvidenceEdge, EvidenceRelation


@dataclass
class ClaimEvidence:
    """Aggregated evidence for a claim."""
    claim: Claim
    supporting_sources: List[Tuple[Source, float]]  # (source, strength)
    contradicting_sources: List[Tuple[Source, float]]
    mentioning_sources: List[Tuple[Source, float]]
    aggregated_confidence: float
    support_count: int
    contradict_count: int


class EvidenceGraph:
    """
    Evidence Graph for claim-source relationships.
    
    Provides:
    - Claim provenance tracking
    - Contradiction detection
    - Confidence aggregation
    - Support scoring
    """
    
    def __init__(self, storage=None):
        """
        Initialize evidence graph.
        
        Args:
            storage: Optional SupabaseStorage for persistence
        """
        self.storage = storage
        
        # In-memory graph for fast operations
        self._claims: Dict[str, Claim] = {}
        self._sources: Dict[str, Source] = {}
        self._edges: Dict[str, EvidenceEdge] = {}
        
        # Adjacency lists for efficient traversal
        self._claim_to_edges: Dict[str, List[str]] = defaultdict(list)
        self._source_to_edges: Dict[str, List[str]] = defaultdict(list)
    
    def add_claim(self, claim: Claim) -> str:
        """Add a claim to the graph."""
        self._claims[claim.id] = claim
        return claim.id
    
    def add_source(self, source: Source) -> str:
        """Add a source to the graph."""
        self._sources[source.id] = source
        return source.id
    
    def add_evidence(
        self,
        claim_id: str,
        source_id: str,
        relation: EvidenceRelation,
        strength: float = 0.5,
        validation_notes: str = "",
    ) -> str:
        """
        Add an evidence edge between a claim and source.
        
        Args:
            claim_id: ID of the claim
            source_id: ID of the source
            relation: Type of relationship
            strength: Strength of the relationship (0-1)
            validation_notes: Notes from validation
            
        Returns:
            Edge ID
        """
        edge = EvidenceEdge(
            from_claim_id=claim_id,
            to_source_id=source_id,
            relation=relation,
            strength=strength,
            validation_notes=validation_notes,
        )
        
        self._edges[edge.id] = edge
        self._claim_to_edges[claim_id].append(edge.id)
        self._source_to_edges[source_id].append(edge.id)
        
        # Update claim's provenance
        if claim_id in self._claims:
            if source_id not in self._claims[claim_id].provenance:
                self._claims[claim_id].provenance.append(source_id)
        
        return edge.id
    
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return self._claims.get(claim_id)
    
    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID."""
        return self._sources.get(source_id)
    
    def get_edges_for_claim(self, claim_id: str) -> List[EvidenceEdge]:
        """Get all edges for a claim."""
        edge_ids = self._claim_to_edges.get(claim_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def get_edges_for_source(self, source_id: str) -> List[EvidenceEdge]:
        """Get all edges pointing to a source."""
        edge_ids = self._source_to_edges.get(source_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def top_supporting_sources(
        self,
        claim_id: str,
        n: int = 5,
    ) -> List[Tuple[Source, float]]:
        """
        Get top sources that support a claim.
        
        Args:
            claim_id: Claim to get support for
            n: Maximum sources to return
            
        Returns:
            List of (Source, strength) tuples, sorted by strength
        """
        edges = self.get_edges_for_claim(claim_id)
        supporting = [
            (self._sources.get(e.to_source_id), e.strength)
            for e in edges
            if e.relation == EvidenceRelation.SUPPORTS and e.to_source_id in self._sources
        ]
        
        # Sort by strength * source reliability
        supporting.sort(
            key=lambda x: x[1] * (x[0].reliability_score if x[0] else 0),
            reverse=True
        )
        
        return [(s, strength) for s, strength in supporting[:n] if s]
    
    def contradictory_claims(
        self,
        claim_id: str,
    ) -> List[Tuple[Claim, Source, float]]:
        """
        Find claims that contradict the given claim.
        
        Looks for sources that:
        1. Support this claim but contradict another
        2. Contradict this claim
        
        Returns:
            List of (contradicting_claim, source, strength) tuples
        """
        # Get sources that contradict this claim
        contradicting_sources = []
        for edge in self.get_edges_for_claim(claim_id):
            if edge.relation == EvidenceRelation.CONTRADICTS:
                source = self._sources.get(edge.to_source_id)
                if source:
                    contradicting_sources.append((source, edge.strength))
        
        # Find other claims supported by contradicting sources
        contradictions = []
        for source, strength in contradicting_sources:
            for edge in self.get_edges_for_source(source.id):
                if edge.from_claim_id != claim_id and edge.relation == EvidenceRelation.SUPPORTS:
                    other_claim = self._claims.get(edge.from_claim_id)
                    if other_claim:
                        contradictions.append((other_claim, source, strength))
        
        return contradictions
    
    def claim_support_score(self, claim_id: str) -> float:
        """
        Calculate aggregated support score for a claim.
        
        Based on:
        - Number and strength of supporting edges
        - Reliability scores of supporting sources
        - Penalty for contradicting edges
        
        Returns:
            Score between 0 and 1
        """
        edges = self.get_edges_for_claim(claim_id)
        
        if not edges:
            return 0.0
        
        support_score = 0.0
        contradict_score = 0.0
        
        for edge in edges:
            source = self._sources.get(edge.to_source_id)
            reliability = source.reliability_score if source else 0.5
            weighted_strength = edge.strength * reliability
            
            if edge.relation == EvidenceRelation.SUPPORTS:
                support_score += weighted_strength
            elif edge.relation == EvidenceRelation.CONTRADICTS:
                contradict_score += weighted_strength
            # MENTIONS edges don't affect score
        
        # Normalize
        total = support_score + contradict_score
        if total == 0:
            return 0.5  # Neutral if only mentions
        
        confidence = support_score / total
        
        # Update claim's confidence
        if claim_id in self._claims:
            self._claims[claim_id].confidence = confidence
        
        return confidence
    
    def get_claim_evidence(self, claim_id: str) -> Optional[ClaimEvidence]:
        """
        Get comprehensive evidence summary for a claim.
        
        Args:
            claim_id: Claim ID
            
        Returns:
            ClaimEvidence object with all evidence details
        """
        claim = self._claims.get(claim_id)
        if not claim:
            return None
        
        edges = self.get_edges_for_claim(claim_id)
        
        supporting = []
        contradicting = []
        mentioning = []
        
        for edge in edges:
            source = self._sources.get(edge.to_source_id)
            if not source:
                continue
            
            pair = (source, edge.strength)
            if edge.relation == EvidenceRelation.SUPPORTS:
                supporting.append(pair)
            elif edge.relation == EvidenceRelation.CONTRADICTS:
                contradicting.append(pair)
            else:
                mentioning.append(pair)
        
        confidence = self.claim_support_score(claim_id)
        
        return ClaimEvidence(
            claim=claim,
            supporting_sources=sorted(supporting, key=lambda x: -x[1]),
            contradicting_sources=sorted(contradicting, key=lambda x: -x[1]),
            mentioning_sources=sorted(mentioning, key=lambda x: -x[1]),
            aggregated_confidence=confidence,
            support_count=len(supporting),
            contradict_count=len(contradicting),
        )
    
    def get_claim_provenance(self, claim_id: str) -> Dict[str, Any]:
        """
        Get full provenance trail for a claim.
        
        Useful for audit and transparency.
        
        Returns:
            Dictionary with claim details and all evidence
        """
        evidence = self.get_claim_evidence(claim_id)
        if not evidence:
            return {}
        
        return {
            "claim_id": claim_id,
            "claim_text": evidence.claim.text,
            "confidence": evidence.aggregated_confidence,
            "supporting_sources": [
                {
                    "source_id": s.id,
                    "url": s.url,
                    "title": s.title,
                    "reliability": s.reliability_score,
                    "strength": strength,
                }
                for s, strength in evidence.supporting_sources
            ],
            "contradicting_sources": [
                {
                    "source_id": s.id,
                    "url": s.url,
                    "title": s.title,
                    "reliability": s.reliability_score,
                    "strength": strength,
                }
                for s, strength in evidence.contradicting_sources
            ],
            "provenance_chain": evidence.claim.provenance,
            "support_count": evidence.support_count,
            "contradict_count": evidence.contradict_count,
        }
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detect all contradictions in the graph.
        
        Returns:
            List of contradiction reports
        """
        contradictions = []
        
        for claim_id, claim in self._claims.items():
            contras = self.contradictory_claims(claim_id)
            if contras:
                contradictions.append({
                    "claim_id": claim_id,
                    "claim_text": claim.text,
                    "contradictions": [
                        {
                            "other_claim_id": other.id,
                            "other_claim_text": other.text,
                            "via_source": source.url,
                            "strength": strength,
                        }
                        for other, source, strength in contras
                    ]
                })
        
        return contradictions
    
    def get_all_claims_with_confidence(self) -> List[Tuple[Claim, float]]:
        """
        Get all claims sorted by confidence.
        
        Returns:
            List of (Claim, confidence) tuples
        """
        results = []
        for claim_id, claim in self._claims.items():
            confidence = self.claim_support_score(claim_id)
            results.append((claim, confidence))
        
        return sorted(results, key=lambda x: -x[1])
    
    def cluster_claims_by_topic(self) -> Dict[str, List[Claim]]:
        """
        Group claims by overlapping sources (topic clustering).
        
        Claims that share sources are likely related.
        
        Returns:
            Dictionary of topic_id -> claims
        """
        # Build source -> claims mapping
        source_claims: Dict[str, List[str]] = defaultdict(list)
        for claim_id, claim in self._claims.items():
            for source_id in claim.provenance:
                source_claims[source_id].append(claim_id)
        
        # Find connected components (claims that share sources)
        visited = set()
        clusters: Dict[str, List[Claim]] = {}
        cluster_id = 0
        
        for claim_id in self._claims:
            if claim_id in visited:
                continue
            
            # BFS to find connected claims
            cluster = []
            queue = [claim_id]
            
            while queue:
                cid = queue.pop(0)
                if cid in visited:
                    continue
                visited.add(cid)
                
                claim = self._claims.get(cid)
                if claim:
                    cluster.append(claim)
                
                # Add claims that share sources
                for source_id in (claim.provenance if claim else []):
                    for related_cid in source_claims.get(source_id, []):
                        if related_cid not in visited:
                            queue.append(related_cid)
            
            if cluster:
                clusters[f"topic_{cluster_id}"] = cluster
                cluster_id += 1
        
        return clusters
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "claims": {cid: c.to_dict() for cid, c in self._claims.items()},
            "sources": {sid: s.to_dict() for sid, s in self._sources.items()},
            "edges": {eid: e.to_dict() for eid, e in self._edges.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], storage=None) -> "EvidenceGraph":
        """Deserialize graph from dictionary."""
        graph = cls(storage=storage)
        
        for claim_data in data.get("claims", {}).values():
            graph._claims[claim_data["id"]] = Claim.from_dict(claim_data)
        
        for source_data in data.get("sources", {}).values():
            graph._sources[source_data["id"]] = Source.from_dict(source_data)
        
        for edge_data in data.get("edges", {}).values():
            edge = EvidenceEdge.from_dict(edge_data)
            graph._edges[edge.id] = edge
            graph._claim_to_edges[edge.from_claim_id].append(edge.id)
            graph._source_to_edges[edge.to_source_id].append(edge.id)
        
        return graph
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        support_count = sum(
            1 for e in self._edges.values()
            if e.relation == EvidenceRelation.SUPPORTS
        )
        contradict_count = sum(
            1 for e in self._edges.values()
            if e.relation == EvidenceRelation.CONTRADICTS
        )
        
        return {
            "claim_count": len(self._claims),
            "source_count": len(self._sources),
            "edge_count": len(self._edges),
            "support_edges": support_count,
            "contradict_edges": contradict_count,
            "mention_edges": len(self._edges) - support_count - contradict_count,
        }
