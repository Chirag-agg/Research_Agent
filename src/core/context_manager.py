"""
Context Manager for Deep Research Agent

Handles long-horizon context management, including:
- Context compaction for LLM token limits
- Persistent memory across research sessions
- Relevance filtering for sub-agent contexts
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ResearchContext:
    """Represents the current state of a research session"""
    query: str
    sub_queries: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    validated_sources: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    iteration: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ContextManager:
    """
    Manages context for long-horizon research tasks.
    
    Key responsibilities:
    - Maintain persistent research state
    - Compact context for LLM token limits
    - Filter relevant context for specialized agents
    """
    
    MAX_CONTEXT_TOKENS = 8000  # Reserve space for response
    
    def __init__(self):
        self.sessions: Dict[str, ResearchContext] = {}
        self.current_session_id: Optional[str] = None
    
    def create_session(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Create a new research session.
        
        Args:
            query: The main research query
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.sessions[session_id] = ResearchContext(query=query)
        self.current_session_id = session_id
        return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> ResearchContext:
        """Get a research session by ID or return current session."""
        sid = session_id or self.current_session_id
        if sid is None or sid not in self.sessions:
            raise ValueError(f"Session not found: {sid}")
        return self.sessions[sid]
    
    def add_finding(
        self,
        content: str,
        source: str,
        agent: str,
        reliability_score: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Add a research finding to the session.
        
        Args:
            content: The finding content
            source: Source URL or reference
            agent: Which agent produced this finding
            reliability_score: Optional score from validation agent
            session_id: Optional session ID (uses current if not specified)
        """
        session = self.get_session(session_id)
        session.findings.append({
            "content": content,
            "source": source,
            "agent": agent,
            "reliability_score": reliability_score,
            "timestamp": datetime.now().isoformat(),
        })
        session.updated_at = datetime.now().isoformat()
    
    def add_knowledge_gap(
        self,
        gap: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Add an identified knowledge gap for further research."""
        session = self.get_session(session_id)
        if gap not in session.knowledge_gaps:
            session.knowledge_gaps.append(gap)
        session.updated_at = datetime.now().isoformat()
    
    def mark_source_validated(
        self,
        finding_index: int,
        reliability_score: float,
        validation_notes: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Mark a finding as validated by the SVA."""
        session = self.get_session(session_id)
        if 0 <= finding_index < len(session.findings):
            session.findings[finding_index]["reliability_score"] = reliability_score
            session.findings[finding_index]["validation_notes"] = validation_notes
            
            # Add to validated sources if score is acceptable
            if reliability_score >= 0.7:
                session.validated_sources.append(session.findings[finding_index])
        session.updated_at = datetime.now().isoformat()
    
    def increment_iteration(self, session_id: Optional[str] = None) -> int:
        """Increment the reflexion loop iteration counter."""
        session = self.get_session(session_id)
        session.iteration += 1
        session.updated_at = datetime.now().isoformat()
        return session.iteration
    
    def get_compact_context(
        self,
        session_id: Optional[str] = None,
        max_findings: int = 10,
        include_gaps: bool = True,
    ) -> str:
        """
        Get a compacted version of the context for LLM consumption.
        
        Implements context engineering by:
        - Limiting number of findings
        - Prioritizing validated sources
        - Summarizing instead of full content
        
        Args:
            session_id: Optional session ID
            max_findings: Maximum findings to include
            include_gaps: Whether to include knowledge gaps
            
        Returns:
            Formatted context string
        """
        session = self.get_session(session_id)
        
        # Prioritize validated sources
        sorted_findings = sorted(
            session.findings,
            key=lambda x: x.get("reliability_score", 0) or 0,
            reverse=True
        )[:max_findings]
        
        context_parts = [
            f"## Research Query\n{session.query}",
            f"\n## Research Iteration: {session.iteration}",
        ]
        
        if session.sub_queries:
            context_parts.append(
                f"\n## Sub-queries\n" + 
                "\n".join(f"- {q}" for q in session.sub_queries)
            )
        
        if sorted_findings:
            findings_text = "\n## Key Findings\n"
            for i, f in enumerate(sorted_findings, 1):
                score = f.get("reliability_score", "unvalidated")
                score_str = f"{score:.2f}" if isinstance(score, float) else score
                findings_text += f"\n### Finding {i} (Reliability: {score_str})\n"
                findings_text += f"Source: {f.get('source', 'Unknown')}\n"
                findings_text += f"Agent: {f.get('agent', 'Unknown')}\n"
                findings_text += f"{f.get('content', '')[:500]}...\n"
            context_parts.append(findings_text)
        
        if include_gaps and session.knowledge_gaps:
            context_parts.append(
                f"\n## Identified Knowledge Gaps\n" +
                "\n".join(f"- {gap}" for gap in session.knowledge_gaps)
            )
        
        return "\n".join(context_parts)
    
    def get_agent_context(
        self,
        agent_type: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Get context tailored for a specific agent type.
        
        Args:
            agent_type: Type of agent (web_search, academic, technical, validator)
            session_id: Optional session ID
            
        Returns:
            Agent-specific context string
        """
        session = self.get_session(session_id)
        
        if agent_type == "web_search":
            return f"Query: {session.query}\nFocus: General web search for current information"
        
        elif agent_type == "academic":
            return f"Query: {session.query}\nFocus: Academic papers, research publications, scholarly sources"
        
        elif agent_type == "technical":
            return f"Query: {session.query}\nFocus: Code repositories, technical documentation, implementation details"
        
        elif agent_type == "validator":
            # Validator needs findings to check
            findings_to_validate = [
                f for f in session.findings
                if f.get("reliability_score") is None
            ]
            return json.dumps({
                "query": session.query,
                "findings_to_validate": findings_to_validate[:5],
            }, indent=2)
        
        else:
            return self.get_compact_context(session_id)
    
    def to_dict(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        session = self.get_session(session_id)
        return {
            "query": session.query,
            "sub_queries": session.sub_queries,
            "findings": session.findings,
            "validated_sources": session.validated_sources,
            "knowledge_gaps": session.knowledge_gaps,
            "iteration": session.iteration,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }
    
    def save_to_file(self, filepath: str, session_id: Optional[str] = None) -> None:
        """Save session to a JSON file."""
        data = self.to_dict(session_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> str:
        """Load session from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = ResearchContext(
            query=data["query"],
            sub_queries=data.get("sub_queries", []),
            findings=data.get("findings", []),
            validated_sources=data.get("validated_sources", []),
            knowledge_gaps=data.get("knowledge_gaps", []),
            iteration=data.get("iteration", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        return session_id
