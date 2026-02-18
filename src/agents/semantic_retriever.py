"""
Deep Research Agent V2 - Semantic Retriever Agent

Memory-augmented retrieval using semantic search over claims and sources.
Combines vector similarity with evidence graph relationships.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseAgent, AgentResult

# V2 imports
try:
    from ..memory.models import Claim, Source
    from ..memory.memory_api import MemoryAPI
    from ..evidence.graph import EvidenceGraph
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    Claim = None
    Source = None
    MemoryAPI = None
    EvidenceGraph = None


@dataclass
class RetrievalResult:
    """Result of semantic retrieval."""
    item_type: str  # "claim" or "source"
    item_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_type": self.item_type,
            "item_id": self.item_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class SemanticRetrieverAgent(BaseAgent):
    """
    Semantic Retriever Agent - Memory-augmented retrieval.
    
    Uses semantic similarity to find:
    1. Related claims from memory
    2. Relevant sources from past research
    3. Cross-session knowledge connections
    """
    
    def __init__(
        self,
        llm_client,
        context_manager=None,
        memory_api: Optional["MemoryAPI"] = None,
        evidence_graph: Optional["EvidenceGraph"] = None,
        embedding_service=None,
    ):
        """
        Initialize the Semantic Retriever.
        
        Args:
            llm_client: LLM client for query enhancement
            context_manager: Optional context manager
            memory_api: V2 memory API for retrieval
            evidence_graph: V2 evidence graph for relationships
            embedding_service: Service for generating embeddings
        """
        super().__init__(llm_client)
        self.context_manager = context_manager
        self.memory_api = memory_api
        self.evidence_graph = evidence_graph
        self.embedding_service = embedding_service
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        prompt = f"""Expand this research query into 3-5 related search queries that would help find relevant information.

QUERY: {query}

Generate variations that:
- Use synonyms
- Focus on different aspects
- Include related concepts

Return as JSON array: ["query1", "query2", ...]"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "Generate query expansions for semantic search."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500,
            )
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            expansions = json.loads(content)
            return [query] + expansions[:4]  # Original + 4 expansions
            
        except Exception as e:
            self.log(f"Query expansion failed: {e}")
            return [query]
    
    async def retrieve_claims(
        self,
        query: str,
        k: int = 10,
        min_confidence: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant claims via semantic search.
        
        Args:
            query: Search query
            k: Number of results
            min_confidence: Minimum claim confidence
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.memory_api or not HAS_V2:
            return []
        
        try:
            # Get embedding for query
            embedding = None
            if self.embedding_service:
                embedding = await self.embedding_service.embed(query)
            
            # Query claims from memory
            claims = await self.memory_api.query_claims(
                embedding=embedding,
                k=k,
                min_confidence=min_confidence,
            )
            
            results = []
            for claim, score in claims:
                results.append(RetrievalResult(
                    item_type="claim",
                    item_id=claim.id,
                    text=claim.text,
                    score=score,
                    metadata={
                        "confidence": claim.confidence,
                        "provenance": claim.provenance,
                        "normalized_text": claim.normalized_text,
                    },
                ))
            
            return results
            
        except Exception as e:
            self.log(f"Claim retrieval failed: {e}")
            return []
    
    async def retrieve_sources(
        self,
        query: str,
        k: int = 10,
        min_reliability: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant sources via semantic search.
        
        Args:
            query: Search query
            k: Number of results
            min_reliability: Minimum source reliability
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.memory_api or not HAS_V2:
            return []
        
        try:
            # Get embedding for query
            embedding = None
            if self.embedding_service:
                embedding = await self.embedding_service.embed(query)
            
            # Query sources from memory
            sources = await self.memory_api.search_similar_sources(
                embedding=embedding,
                k=k,
            )
            
            results = []
            for source, score in sources:
                if source.reliability_score >= min_reliability:
                    results.append(RetrievalResult(
                        item_type="source",
                        item_id=source.id,
                        text=source.text_excerpt,
                        score=score,
                        metadata={
                            "url": source.url,
                            "title": source.title,
                            "reliability_score": source.reliability_score,
                            "domain": source.domain,
                        },
                    ))
            
            return results
            
        except Exception as e:
            self.log(f"Source retrieval failed: {e}")
            return []
    
    async def retrieve_with_evidence(
        self,
        query: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve claims with their evidence graph connections.
        
        Args:
            query: Search query
            k: Number of claims to retrieve
            
        Returns:
            Dictionary with claims and their evidence
        """
        if not self.evidence_graph or not HAS_V2:
            return {"claims": [], "evidence": []}
        
        # Get relevant claims
        claim_results = await self.retrieve_claims(query, k=k)
        
        # Get evidence for each claim
        claims_with_evidence = []
        for result in claim_results:
            evidence = self.evidence_graph.get_claim_provenance(result.item_id)
            claims_with_evidence.append({
                "claim": result.to_dict(),
                "evidence": evidence,
            })
        
        return {
            "claims": claims_with_evidence,
            "total": len(claims_with_evidence),
        }
    
    async def retrieve_related(
        self,
        item_id: str,
        item_type: str = "claim",
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Find items related to a given claim or source.
        
        Uses evidence graph relationships and embedding similarity.
        
        Args:
            item_id: ID of the item
            item_type: "claim" or "source"
            k: Number of results
            
        Returns:
            List of related items
        """
        if not self.evidence_graph or not HAS_V2:
            return []
        
        related = []
        
        if item_type == "claim":
            # Get claim from graph
            claim = self.evidence_graph.get_claim(item_id)
            if not claim:
                return []
            
            # Find sources that support this claim
            for source_id in claim.provenance:
                source = self.evidence_graph.get_source(source_id)
                if source:
                    edges = self.evidence_graph.get_edges_for_source(source_id)
                    score = sum(e.strength for e in edges) / max(len(edges), 1)
                    
                    related.append(RetrievalResult(
                        item_type="source",
                        item_id=source.id,
                        text=source.text_excerpt,
                        score=score,
                        metadata={
                            "url": source.url,
                            "title": source.title,
                            "relation": "supports",
                        },
                    ))
            
            # Find contradicting claims
            contradictions = self.evidence_graph.contradictory_claims(item_id)
            for contra_claim, via_source, strength in contradictions:
                related.append(RetrievalResult(
                    item_type="claim",
                    item_id=contra_claim.id,
                    text=contra_claim.text,
                    score=strength,
                    metadata={
                        "relation": "contradicts",
                        "via_source": via_source.url if via_source else None,
                    },
                ))
        
        elif item_type == "source":
            # Get all claims linked to this source
            edges = self.evidence_graph.get_edges_for_source(item_id)
            for edge in edges[:k]:
                claim = self.evidence_graph.get_claim(edge.from_claim_id)
                if claim:
                    related.append(RetrievalResult(
                        item_type="claim",
                        item_id=claim.id,
                        text=claim.text,
                        score=edge.strength,
                        metadata={
                            "relation": edge.relation.value,
                            "confidence": claim.confidence,
                        },
                    ))
        
        # Sort by score and limit
        related.sort(key=lambda r: -r.score)
        return related[:k]
    
    async def recall_from_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant information from Mem0 memory.
        
        Args:
            query: Search query
            user_id: Optional user ID for personalized memory
            
        Returns:
            List of memory records
        """
        if not self.memory_api or not HAS_V2:
            return []
        
        try:
            memories = await self.memory_api.recall_memories(
                query=query,
                user_id=user_id,
            )
            return memories
        except Exception as e:
            self.log(f"Memory recall failed: {e}")
            return []
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute semantic retrieval.
        
        Args:
            input_data: Must contain 'query', optional 'retrieval_type'
            
        Returns:
            AgentResult with retrieved items
        """
        query = input_data.get("query", "")
        retrieval_type = input_data.get("retrieval_type", "all")  # claims, sources, all
        k = input_data.get("k", 10)
        expand_query = input_data.get("expand_query", False)
        
        if not query:
            return AgentResult(
                success=False,
                content=None,
                agent_name="SemanticRetrieverAgent",
                error="No query provided",
            )
        
        try:
            # Expand query if requested
            queries = [query]
            if expand_query:
                queries = self.expand_query(query)
            
            # Note: In production, these would be async
            # For now, return the structure without actual retrieval
            # since embedding service may not be configured
            
            result = {
                "queries": queries,
                "claims": [],
                "sources": [],
                "memories": [],
                "message": "Retrieval requires configured memory_api and embedding_service",
            }
            
            return AgentResult(
                success=True,
                content=result,
                agent_name="SemanticRetrieverAgent",
                metadata={
                    "retrieval_type": retrieval_type,
                    "k": k,
                    "queries_used": len(queries),
                },
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name="SemanticRetrieverAgent",
                error=str(e),
            )
