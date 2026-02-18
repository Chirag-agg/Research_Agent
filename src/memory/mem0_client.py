"""
Deep Research Agent V2 - Mem0 Memory Client

Integrates with Mem0 for intelligent memory management:
- Automatic memory extraction from conversations
- Semantic search over memories
- Memory lifecycle management
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Memory:
    """A memory item from Mem0."""
    id: str
    memory: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    categories: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data.get("id", ""),
            memory=data.get("memory", ""),
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata"),
            categories=data.get("categories"),
        )


class Mem0Client:
    """
    Client for Mem0 memory management API.
    
    Provides memory operations for the Deep Research Agent:
    - Store research findings as memories
    - Search relevant memories for new queries
    - Track claim-level knowledge across sessions
    """
    
    BASE_URL = "https://api.mem0.ai"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """
        Initialize Mem0 client.
        
        Args:
            api_key: Mem0 API key (or MEM0_API_KEY env var)
            org_id: Organization ID (optional)
            project_id: Project ID (optional)
        """
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self.org_id = org_id or os.getenv("MEM0_ORG_ID")
        self.project_id = project_id or os.getenv("MEM0_PROJECT_ID")
        
        if not self.api_key:
            raise ValueError("MEM0_API_KEY must be set")
        
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def add_memory(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        categories: Optional[Dict[str, str]] = None,
        infer: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Add memories from conversation messages.
        
        Args:
            messages: List of {"role": "user|assistant", "content": "..."} messages
            user_id: User identifier for memory association
            agent_id: Agent identifier (e.g., "research-agent")
            run_id: Run/session identifier
            metadata: Additional metadata (source_ids, claim_ids, etc.)
            categories: Custom categories with descriptions
            infer: Whether to extract memories or store raw
            
        Returns:
            List of created memory events with ids
        """
        payload = {
            "messages": messages,
            "infer": infer,
            "output_format": "v1.1",
            "version": "v2",
        }
        
        if user_id:
            payload["user_id"] = user_id
        if agent_id:
            payload["agent_id"] = agent_id
        if run_id:
            payload["run_id"] = run_id
        if metadata:
            payload["metadata"] = metadata
        if categories:
            payload["custom_categories"] = categories
        if self.org_id:
            payload["org_id"] = self.org_id
        if self.project_id:
            payload["project_id"] = self.project_id
        
        response = requests.post(
            f"{self.BASE_URL}/v1/memories/",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("results", result)
    
    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        threshold: float = 0.7,
        rerank: bool = True,
    ) -> List[Memory]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            filters: Advanced filter conditions
            top_k: Number of results
            threshold: Minimum similarity threshold
            rerank: Whether to rerank results
            
        Returns:
            List of matching Memory objects
        """
        payload = {
            "query": query,
            "version": "v2",
            "top_k": top_k,
            "threshold": threshold,
            "rerank": rerank,
        }
        
        # Build filters
        filter_conditions = filters or {}
        if user_id:
            filter_conditions["user_id"] = user_id
        if agent_id:
            filter_conditions["agent_id"] = agent_id
        
        if filter_conditions:
            payload["filters"] = filter_conditions
        
        if self.org_id:
            payload["org_id"] = self.org_id
        if self.project_id:
            payload["project_id"] = self.project_id
        
        response = requests.post(
            f"{self.BASE_URL}/v1/memories/search/",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        
        result = response.json()
        memories_data = result.get("results", result)
        
        return [Memory.from_dict(m) for m in memories_data]
    
    def get_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """
        Get all memories with optional filters.
        
        Args:
            user_id: Filter by user
            agent_id: Filter by agent
            run_id: Filter by run/session
            limit: Maximum number of memories
            
        Returns:
            List of Memory objects
        """
        params = {"limit": limit}
        
        if user_id:
            params["user_id"] = user_id
        if agent_id:
            params["agent_id"] = agent_id
        if run_id:
            params["run_id"] = run_id
        if self.org_id:
            params["org_id"] = self.org_id
        if self.project_id:
            params["project_id"] = self.project_id
        
        response = requests.get(
            f"{self.BASE_URL}/v1/memories/",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        
        result = response.json()
        memories_data = result.get("results", result)
        
        return [Memory.from_dict(m) for m in memories_data]
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        response = requests.get(
            f"{self.BASE_URL}/v1/memories/{memory_id}/",
            headers=self.headers,
            timeout=30,
        )
        
        if response.status_code == 404:
            return None
        
        response.raise_for_status()
        return Memory.from_dict(response.json())
    
    def update_memory(
        self,
        memory_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of memory to update
            text: New memory text
            
        Returns:
            Update response
        """
        payload = {"text": text}
        
        response = requests.put(
            f"{self.BASE_URL}/v1/memories/{memory_id}/",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted successfully
        """
        response = requests.delete(
            f"{self.BASE_URL}/v1/memories/{memory_id}/",
            headers=self.headers,
            timeout=30,
        )
        return response.status_code == 200
    
    def delete_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Delete all memories for a user or agent.
        
        Args:
            user_id: Delete memories for this user
            agent_id: Delete memories for this agent
            
        Returns:
            True if deleted successfully
        """
        params = {}
        if user_id:
            params["user_id"] = user_id
        if agent_id:
            params["agent_id"] = agent_id
        
        response = requests.delete(
            f"{self.BASE_URL}/v1/memories/",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        return response.status_code == 200


class ResearchMemoryManager:
    """
    High-level memory manager for research workflows.
    
    Coordinates between Mem0 (semantic memory) and Supabase (structured data)
    to provide a unified memory interface for the research agent.
    """
    
    def __init__(
        self,
        mem0_client: Mem0Client,
        supabase_storage: Optional[Any] = None,  # SupabaseStorage
    ):
        """
        Initialize research memory manager.
        
        Args:
            mem0_client: Mem0 client for semantic memory
            supabase_storage: Supabase storage for structured data
        """
        self.mem0 = mem0_client
        self.storage = supabase_storage
        self.agent_id = "deep-research-agent"
    
    def store_research_findings(
        self,
        session_id: str,
        query: str,
        findings: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store research findings as memories.
        
        Args:
            session_id: Research session ID
            query: Original research query
            findings: List of validated findings
            user_id: User who initiated research
            
        Returns:
            List of created memory IDs
        """
        memory_ids = []
        
        # Convert findings to conversation format for Mem0
        messages = [
            {"role": "user", "content": f"Research query: {query}"}
        ]
        
        for finding in findings:
            content = f"""
Research finding from {finding.get('source', 'unknown source')}:
{finding.get('content', finding.get('text', ''))}

Key claims:
{json.dumps(finding.get('claims', []), indent=2)}

Reliability score: {finding.get('reliability_score', 'N/A')}
"""
            messages.append({"role": "assistant", "content": content})
        
        # Store via Mem0 with research-specific categories
        result = self.mem0.add_memory(
            messages=messages,
            user_id=user_id,
            agent_id=self.agent_id,
            run_id=session_id,
            metadata={
                "session_id": session_id,
                "query": query,
                "finding_count": len(findings),
                "type": "research_findings",
            },
            categories={
                "research": "Research findings and claims",
                "sources": "Information about sources",
                "claims": "Factual claims extracted from research",
            },
        )
        
        for item in result:
            if item.get("event") == "ADD" and item.get("id"):
                memory_ids.append(item["id"])
        
        return memory_ids
    
    def recall_relevant_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Memory]:
        """
        Recall memories relevant to a new research query.
        
        Useful for:
        - Avoiding redundant research
        - Building on previous findings
        - Identifying related claims
        
        Args:
            query: New research query
            user_id: User context
            top_k: Number of memories to return
            
        Returns:
            List of relevant memories
        """
        return self.mem0.search_memories(
            query=query,
            user_id=user_id,
            agent_id=self.agent_id,
            top_k=top_k,
            threshold=0.6,  # Lower threshold for broad recall
            rerank=True,
        )
    
    def get_session_memories(self, session_id: str) -> List[Memory]:
        """Get all memories from a specific research session."""
        return self.mem0.get_memories(
            run_id=session_id,
            agent_id=self.agent_id,
        )
    
    def store_claim(
        self,
        claim_text: str,
        source_url: str,
        confidence: float,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a specific claim as a memory.
        
        Args:
            claim_text: The claim statement
            source_url: URL of the source
            confidence: Confidence score
            session_id: Session ID
            user_id: User ID
            
        Returns:
            Memory ID if created
        """
        messages = [
            {
                "role": "assistant",
                "content": f"Verified claim (confidence: {confidence:.2f}): {claim_text}\nSource: {source_url}"
            }
        ]
        
        result = self.mem0.add_memory(
            messages=messages,
            user_id=user_id,
            agent_id=self.agent_id,
            run_id=session_id,
            metadata={
                "type": "claim",
                "source_url": source_url,
                "confidence": confidence,
            },
            infer=False,  # Store exactly as provided
        )
        
        for item in result:
            if item.get("id"):
                return item["id"]
        
        return None
