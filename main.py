"""
Deep Research Agent V2 - Main Orchestrator

The V2 orchestrator integrates:
- Hierarchical planning with task graphs
- Memory persistence via Supabase + Mem0
- Evidence graph for claim traceability
- Parallel task execution

This replaces the original DeepResearchOrchestrator with enhanced capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Core imports
from src.core.llm_client import LLMClient
from src.core.context_manager import ContextManager

# Agent imports
from src.agents.master_planner import MasterPlannerAgent
from src.agents.web_search import WebSearchAgent
from src.agents.source_validator import SourceValidatorAgent
from src.agents.reflexion import ReflexionAgent

# V2 imports
from src.agents.hierarchical_planner import HierarchicalPlannerAgent
from src.agents.claim_extractor import ClaimExtractorAgent
from src.planning.task_graph import TaskGraph, TaskType, TaskStatus
from src.planning.executor import TaskExecutor
from src.memory.models import Claim, Source, Session, EvidenceRelation
from src.memory.memory_api import MemoryAPI
from src.evidence.graph import EvidenceGraph
from src.planning.task_graph import TaskGraph

load_dotenv()


@dataclass
class ResearchResultV2:
    """Enhanced research result with claim provenance."""
    query: str
    report: str
    sources: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]
    evidence_graph: Dict[str, Any]
    metadata: Dict[str, Any]
    iterations: int
    task_graph: Dict[str, Any]
    execution_stats: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DeepResearchOrchestratorV2:
    """
    V2 Orchestrator for the Deep Research Agent.
    
    Key improvements over V1:
    - Hierarchical task planning with dynamic replanning
    - Memory persistence via Supabase + Mem0  
    - Evidence graph for claim-source relationships
    - Parallel task execution with budget management
    """
    
    def __init__(
        self,
        search_provider: str = "exa",
        max_iterations: int = 3,
        verbose: bool = True,
        use_memory: bool = True,
    ):
        """
        Initialize the V2 orchestrator.
        
        Args:
            search_provider: Search API to use (exa, tavily)
            max_iterations: Maximum reflexion iterations
            verbose: Enable verbose logging
            use_memory: Enable memory persistence
        """
        self.search_provider = search_provider
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Initialize context manager
        self.context_manager = ContextManager()
        
        # Initialize memory API if enabled
        self.memory_api = None
        if use_memory:
            try:
                self.memory_api = MemoryAPI()
                self.log("Memory API initialized")
            except Exception as e:
                self.log(f"Memory API not available: {e}")
        
        # Initialize evidence graph
        self.evidence_graph = EvidenceGraph()
        
        # Initialize agents
        self._init_agents()
        
        # Initialize task executor
        self.executor = TaskExecutor(
            agents=self.agents,
            memory_api=self.memory_api,
            max_workers=5,
        )
    
    def _init_agents(self):
        """Initialize all agents."""
        # V1 agents (still used)
        self.master_planner = MasterPlannerAgent(
            self.llm_client, 
            context_manager=self.context_manager
        )
        self.web_search = WebSearchAgent(
            self.llm_client,
            context_manager=self.context_manager,
            search_provider=self.search_provider,
        )
        self.source_validator = SourceValidatorAgent(
            self.llm_client,
            context_manager=self.context_manager,
            evidence_graph=self.evidence_graph,  # V2: Pass evidence graph
            memory_api=self.memory_api,
        )
        
        # V2 agents
        self.hierarchical_planner = HierarchicalPlannerAgent(
            self.llm_client,
            context_manager=self.context_manager,
            memory_api=self.memory_api,
        )
        
        self.reflexion = ReflexionAgent(
            self.llm_client,
            context_manager=self.context_manager,
            planner=self.hierarchical_planner,  # V2: Pass planner for replanning
        )
        self.claim_extractor = ClaimExtractorAgent(
            self.llm_client,
            context_manager=self.context_manager,
            memory_api=self.memory_api,
        )
        
        # Agent registry for executor
        self.agents = {
            "MasterPlannerAgent": self.master_planner,
            "WebSearchAgent": self.web_search,
            "AcademicSearchAgent": self.web_search,  # Fallback to web search
            "TechnicalSearchAgent": self.web_search,  # Fallback to web search
            "SourceValidatorAgent": self.source_validator,
            "ReflexionAgent": self.reflexion,
            "HierarchicalPlannerAgent": self.hierarchical_planner,
            "ClaimExtractorAgent": self.claim_extractor,
        }
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[V2 Orchestrator] {message}")

    def _compute_task_graph_stats(self, task_graph: TaskGraph) -> Dict[str, int]:
        """Compute total nodes and max dependency depth for a task graph."""
        total_nodes = len(task_graph.nodes)
        depth_cache: Dict[str, int] = {}

        def depth(node_id: str, visiting: Optional[set] = None) -> int:
            if node_id in depth_cache:
                return depth_cache[node_id]
            if visiting is None:
                visiting = set()
            if node_id in visiting:
                # Break cycles defensively
                return 1
            visiting.add(node_id)

            node = task_graph.nodes.get(node_id)
            if not node or not node.dependencies:
                depth_cache[node_id] = 1
                return 1

            dep_depths = [depth(dep_id, visiting) for dep_id in node.dependencies]
            max_depth = 1 + (max(dep_depths) if dep_depths else 0)
            depth_cache[node_id] = max_depth
            return max_depth

        max_depth = 0
        for node_id in task_graph.nodes:
            max_depth = max(max_depth, depth(node_id))

        return {"total_nodes": total_nodes, "max_depth": max_depth}
    
    async def research_async(
        self, 
        query: str, 
        preferences: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        previous_context: Optional[Dict[str, Any]] = None,
    ) -> ResearchResultV2:
        """
        Execute a full V2 research workflow asynchronously.
        
        Args:
            query: Research query
            preferences: Optional preferences (depth, time_budget, etc.)
            session_id: Optional session ID to allow external management
            
        Returns:
            ResearchResultV2 with complete results
        """
        prefs = preferences or {}
        mode = prefs.get("mode", "deep")
        quick_mode = mode == "quick"
        self.log(f"Starting V2 research: mode={mode} query={query}")

        start_time = time.time()
        usage_before = self.llm_client.get_usage_stats()

        # Configure time and retrieval defaults per mode
        if quick_mode:
            prefs.setdefault("max_time_ms", 60000)
            prefs.setdefault("max_iterations", 1)
            prefs.setdefault("min_sources", 3)
            prefs.setdefault("top_k", 3)
            # Temporarily bias model tiers toward fast model
            fast_model = self.llm_client.MODELS.get("fast", self.llm_client.MODELS.get("default"))
            _orig_tiers = {k: list(v) for k, v in self.llm_client.MODEL_TIERS.items()}
            self.llm_client.MODEL_TIERS = {
                "small": [fast_model],
                "medium": [fast_model],
                "large": [fast_model],
            }
        else:
            prefs.setdefault("max_time_ms", 480000)
            prefs.setdefault("max_iterations", 3)
            prefs.setdefault("min_sources", 6)
            prefs.setdefault("top_k", 8)
        
        try:
            # Create session
            session = None
            if self.memory_api:
                session = await self.memory_api.create_session(
                    query=query,
                    user_id=prefs.get("user_id"),
                    session_id=session_id
                )
                self.log(f"Created session: {session.id}")
            
            # Load prior context if provided
            prior_messages = []
            prior_graph = None
            prior_task_graph = None
            prior_summary = None

            if previous_context:
                prior_messages = previous_context.get("messages", [])
                prior_summary = previous_context.get("summary")
                if previous_context.get("evidence_graph"):
                    try:
                        prior_graph = EvidenceGraph.from_dict(previous_context.get("evidence_graph"))
                        self.evidence_graph = prior_graph
                    except Exception as e:
                        self.log(f"Failed to load prior evidence graph: {e}")
                if previous_context.get("task_graph"):
                    try:
                        prior_task_graph = TaskGraph.from_dict(previous_context.get("task_graph"))
                    except Exception as e:
                        self.log(f"Failed to load prior task graph: {e}")

            # Generate task graph using hierarchical planner, optionally conditioned on prior context
            self.log("Generating task graph...")
            task_graph = prior_task_graph or self.hierarchical_planner.hierarchical_plan(
                query=query,
                preferences=prefs,
                session_id=session.id if session else None,
                previous_summary=prior_summary,
                previous_messages=prior_messages,
            )
            self.log(f"Task graph created with {len(task_graph.nodes)} nodes")
            
            # Execute task graph with prior context injected so follow-ups refine
            self.log("Executing task graph...")
            context = await self.executor.execute_graph(
                task_graph=task_graph,
                initial_context={
                    "query": query,
                    "session_id": session.id if session else None,
                    "mode": mode,
                    "previous_summary": prior_summary,
                    "previous_messages": prior_messages,
                    "previous_evidence_graph": previous_context.get("evidence_graph") if previous_context else None,
                },
            )
            self.log(f"Execution complete: {context.get('execution_stats', {})}")
            
            # Check if replanning is needed (skip in quick mode)
            reflexion_result = context.get("reflexion_result")
            reflexion_triggered = False
            if not quick_mode and reflexion_result and reflexion_result.content and reflexion_result.content.get("replan_required", False):
                self.log("Replanning based on reflexion feedback...")
                reflexion_triggered = True
                task_graph = self.hierarchical_planner.replan(
                    task_graph=task_graph,
                    reflexion_feedback=reflexion_result.content,
                    preferences=prefs,
                )
                
                # Re-execute with updated graph
                context = await self.executor.execute_graph(
                    task_graph=task_graph,
                    initial_context=context,
                )
            
            # Build or merge evidence graph from findings
            self.log("Building evidence graph...")
            await self._build_evidence_graph(context)
            
            # Extract claims
            self.log("Extracting claims...")
            claims = context.get("claims", [])
            sources = context.get("validated_findings", [])
            if not sources:
                if quick_mode:
                    self.log("Quick mode: validation skipped, using raw findings")
                else:
                    self.log("No validated findings found, falling back to raw findings")
                sources = context.get("findings", [])
            
            # Generate final report
            self.log("Synthesizing report...")
            report = await self._synthesize_report(query, claims, sources)
            
            # Store memories if enabled
            if self.memory_api and session:
                self.log("Storing memories...")
                self.memory_api.store_research_findings(
                    session_id=session.id,
                    query=query,
                    findings=sources,
                    user_id=prefs.get("user_id"),
                )
            
            # Build result
            usage_after = self.llm_client.get_usage_stats()
            latency_seconds = max(0.0, time.time() - start_time)
            prompt_tokens = usage_after.get("prompt_tokens", 0) - usage_before.get("prompt_tokens", 0)
            completion_tokens = usage_after.get("completion_tokens", 0) - usage_before.get("completion_tokens", 0)
            estimated_cost = usage_after.get("estimated_cost_usd", 0.0) - usage_before.get("estimated_cost_usd", 0.0)
            model_usage_before = usage_before.get("model_usage_breakdown", {})
            model_usage_after = usage_after.get("model_usage_breakdown", {})
            model_usage_breakdown = {
                provider: model_usage_after.get(provider, 0) - model_usage_before.get(provider, 0)
                for provider in model_usage_after
            }
            task_graph_stats = self._compute_task_graph_stats(task_graph)

            result = ResearchResultV2(
                query=query,
                report=report,
                sources=[self._format_source(s) for s in sources],
                claims=claims,
                evidence_graph=self.evidence_graph.to_dict(),
                metadata={
                    "search_provider": self.search_provider,
                    "session_id": session.id if session else None,
                    "metrics": {
                        "mode": mode,
                        "latency": latency_seconds,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_estimate": estimated_cost,
                        "models_used": model_usage_breakdown,
                        "task_graph": task_graph_stats,
                    },
                    "reflexion": {
                        "triggered": reflexion_triggered,
                    },
                },
                iterations=task_graph.current_iteration,
                task_graph=task_graph.to_dict(),
                execution_stats=context.get("execution_stats", {}),
            )
            
            self.log("Research complete!")
            return result
            
        except Exception as e:
            self.log(f"Research failed: {e}")
            return self._error_result(query, str(e))
        finally:
            if quick_mode:
                # Restore original model tiers if we overrode them for quick mode
                self.llm_client.MODEL_TIERS = _orig_tiers
    
    def research(
        self,
        query: str,
        preferences: Optional[Dict[str, Any]] = None,
        previous_context: Optional[Dict[str, Any]] = None,
    ) -> ResearchResultV2:
        """
        Execute research synchronously (wrapper for async).
        
        Args:
            query: Research query
            preferences: Optional preferences
            
        Returns:
            ResearchResultV2
        """
        return asyncio.run(self.research_async(query, preferences, previous_context=previous_context))
    
    async def _build_evidence_graph(self, context: Dict[str, Any]) -> None:
        """Build evidence graph from context."""
        sources = context.get("validated_findings", [])
        claims = context.get("claims", [])
        
        # Add sources to graph
        for source_data in sources:
            source = Source(
                url=source_data.get("url", source_data.get("source", "")),
                title=source_data.get("title", ""),
                text_excerpt=source_data.get("content", source_data.get("text", ""))[:2000],
                reliability_score=source_data.get("reliability_score", 0.5),
            )
            self.evidence_graph.add_source(source)
            source_data["id"] = source.id
        
        # Add claims and edges
        for claim_data in claims:
            claim = Claim(
                text=claim_data.get("claim", claim_data.get("text", "")),
                normalized_text=claim_data.get("normalized_text", ""),
                confidence=claim_data.get("confidence", 0.5),
                supporting_text=claim_data.get("supporting_text", ""),
            )
            self.evidence_graph.add_claim(claim)
            
            # Create edge to source
            source_id = claim_data.get("source_id")
            if source_id:
                relation = (
                    EvidenceRelation.SUPPORTS
                    if claim_data.get("confidence", 0.5) > 0.6
                    else EvidenceRelation.MENTIONS
                )
                self.evidence_graph.add_evidence(
                    claim_id=claim.id,
                    source_id=source_id,
                    relation=relation,
                    strength=claim_data.get("confidence", 0.5),
                )
    
    async def _synthesize_report(
        self,
        query: str,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> str:
        """Generate the final research report."""
        # Get claim provenance from evidence graph
        claims_with_evidence = []
        for claim_data in claims:
            evidence = self.evidence_graph.get_claim_provenance(claim_data.get("id", ""))
            if evidence:
                claims_with_evidence.append(evidence)
            else:
                claims_with_evidence.append(claim_data)
        
        # Use master planner for synthesis
        result = self.master_planner.synthesize_findings(
            query=query,
            findings=sources,
            format_type="markdown",
        )
        
        # Add evidence section
        evidence_section = self._format_evidence_section(claims_with_evidence)
        
        if isinstance(result, str):
            return result + "\n\n" + evidence_section
        return result.get("report", str(result)) + "\n\n" + evidence_section
    
    def _format_evidence_section(self, claims: List[Dict[str, Any]]) -> str:
        """Format the evidence trail section."""
        if not claims:
            return ""
        
        lines = ["\n## Evidence Trail\n"]
        lines.append("The following claims are supported by the evidence graph:\n")
        
        for i, claim in enumerate(claims[:10], 1):  # Limit to 10
            claim_text = claim.get("claim_text", claim.get("claim", ""))
            confidence = claim.get("confidence", claim.get("aggregated_confidence", 0))
            
            lines.append(f"\n### Claim {i}")
            lines.append(f"> {claim_text}")
            lines.append(f"\n**Confidence:** {confidence:.2f}")
            
            supporting = claim.get("supporting_sources", [])
            if supporting:
                lines.append("\n**Supporting Sources:**")
                for source in supporting[:3]:
                    if isinstance(source, dict):
                        lines.append(f"- [{source.get('title', 'Source')}]({source.get('url', '')})")
        
        return "\n".join(lines)
    
    def _format_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Format source for output."""
        return {
            "url": source.get("url", source.get("source", "")),
            "title": source.get("title", ""),
            "content": source.get("content", source.get("text", ""))[:500],
            "reliability_score": source.get("reliability_score", 0.5),
            "id": source.get("id", ""),
        }
    
    def _error_result(self, query: str, error: str) -> ResearchResultV2:
        """Generate an error result."""
        return ResearchResultV2(
            query=query,
            report=f"# Research Failed\n\nError: {error}",
            sources=[],
            claims=[],
            evidence_graph={},
            metadata={"error": error},
            iterations=0,
            task_graph={},
            execution_stats={},
        )
    
    def save_result(self, result: ResearchResultV2, filepath: str) -> None:
        """Save research result to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "query": result.query,
                "report": result.report,
                "sources": result.sources,
                "claims": result.claims,
                "evidence_graph": result.evidence_graph,
                "metadata": result.metadata,
                "iterations": result.iterations,
                "task_graph": result.task_graph,
                "execution_stats": result.execution_stats,
                "created_at": result.created_at,
            }, f, indent=2, default=str)
        self.log(f"Result saved to {filepath}")
    
    def shutdown(self):
        """Shutdown the orchestrator."""
        self.executor.shutdown()


def main():
    """Run a sample V2 research query."""
    print("=" * 60)
    print("Deep Research Agent V2")
    print("=" * 60)
    
    orchestrator = DeepResearchOrchestratorV2(
        search_provider="exa",
        max_iterations=3,
        verbose=True,
        use_memory=False,  # Set to True when Mem0/Supabase configured
    )
    
    query = "What are the latest advancements in AI agent architectures in 2024?"
    
    print(f"\nResearch Query: {query}\n")
    print("-" * 60)
    
    result = orchestrator.research(query)
    
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(result.report)
    
    print("\n" + "-" * 60)
    print("EXECUTION STATS")
    print("-" * 60)
    print(json.dumps(result.execution_stats, indent=2))
    
    print("\n" + "-" * 60)
    print(f"Claims extracted: {len(result.claims)}")
    print(f"Sources used: {len(result.sources)}")
    print(f"Iterations: {result.iterations}")
    
    # Save result
    orchestrator.save_result(result, "research_result_v2.json")
    
    orchestrator.shutdown()


if __name__ == "__main__":
    main()
