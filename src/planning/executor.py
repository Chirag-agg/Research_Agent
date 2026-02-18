"""
Deep Research Agent V2 - Task Executor

Parallel execution engine for task graphs.
Routes tasks to appropriate agents and manages execution flow.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
from concurrent.futures import ThreadPoolExecutor

from .task_graph import TaskGraph, TaskNode, TaskType, TaskStatus, ModelTier


class TaskExecutor:
    """
    Parallel Task Executor for research workflows.
    
    Responsibilities:
    - Execute ready tasks in parallel
    - Route tasks to appropriate agents
    - Track execution status and results
    - Handle failures and retries
    """
    
    def __init__(
        self,
        agents: Dict[str, Any],
        memory_api: Optional[Any] = None,
        max_workers: int = 5,
    ):
        """
        Initialize the task executor.
        
        Args:
            agents: Dictionary of agent_name -> agent_instance
            memory_api: Optional MemoryAPI for result storage
            max_workers: Maximum parallel tasks
        """
        self.agents = agents
        self.memory_api = memory_api
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task type to agent mapping
        self.type_to_agent = {
            TaskType.SEARCH_WEB: "WebSearchAgent",
            TaskType.SEARCH_ACADEMIC: "AcademicSearchAgent",
            TaskType.SEARCH_TECHNICAL: "TechnicalSearchAgent",
            TaskType.SEARCH_CITATION: "CitationCrawlerAgent",
            TaskType.EXTRACT_CLAIMS: "ClaimExtractorAgent",
            TaskType.VALIDATE_CLAIMS: "SourceValidatorAgent",
            TaskType.REFLEXION: "ReflexionAgent",
            TaskType.SYNTHESIZE_REPORT: "MasterPlannerAgent",
            TaskType.MERGE_EVIDENCE: None,  # Internal operation
            TaskType.DEDUPLICATE_CLAIMS: None,
        }
    
    def route_to_agent(self, node: TaskNode) -> Optional[Any]:
        """
        Select appropriate agent for a task.
        
        Args:
            node: Task node to execute
            
        Returns:
            Agent instance or None
        """
        # Use preferred agent if specified and available
        if node.preferred_agent and node.preferred_agent in self.agents:
            return self.agents[node.preferred_agent]
        
        # Fall back to type-based routing
        agent_name = self.type_to_agent.get(node.type)
        if agent_name and agent_name in self.agents:
            return self.agents[agent_name]
        
        return None
    
    def _execute_node_sync(
        self,
        node: TaskNode,
        context: Dict[str, Any],
    ) -> Any:
        """
        Execute a single node synchronously.
        
        Args:
            node: Task node to execute
            context: Execution context with accumulated results
            
        Returns:
            Task result
        """
        agent = self.route_to_agent(node)
        
        if not agent:
            # Handle internal operations without agents
            if node.type == TaskType.MERGE_EVIDENCE:
                return self._merge_evidence(node, context)
            elif node.type == TaskType.DEDUPLICATE_CLAIMS:
                return self._deduplicate_claims(node, context)
            else:
                raise ValueError(f"No agent available for task type: {node.type}")
        
        # Prepare input data from node input and dependency results
        input_data = dict(node.input)
        
        # Add results from dependencies
        for dep_id in node.dependencies:
            dep_result = context.get("results", {}).get(dep_id)
            if dep_result:
                input_data[f"dep_{dep_id}"] = dep_result
        
        # Add accumulated findings for validation/synthesis
        if node.type in [TaskType.VALIDATE_CLAIMS, TaskType.EXTRACT_CLAIMS]:
            input_data["sources"] = context.get("sources", [])
            input_data["findings"] = context.get("findings", [])
        
        if node.type in [TaskType.REFLEXION, TaskType.SYNTHESIZE_REPORT]:
            input_data["findings"] = context.get("validated_findings", [])
            input_data["claims"] = context.get("claims", [])

        if "mode" in context:
            input_data["mode"] = context.get("mode")
        
        # Execute agent
        result = agent.execute(input_data)
        
        return result
    
    async def execute_node(
        self,
        node: TaskNode,
        task_graph: TaskGraph,
        context: Dict[str, Any],
    ) -> Any:
        """
        Execute a node asynchronously.
        
        Args:
            node: Task node
            task_graph: Parent task graph
            context: Execution context
            
        Returns:
            Task result
        """
        # Mark as running
        task_graph.mark_running(node.id)
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_node_sync,
                node,
                context,
            )
            
            # Mark complete
            task_graph.mark_complete(node.id, result)
            
            # Update context with results
            self._update_context(node, result, context)
            
            return result
            
        except Exception as e:
            # Mark failed
            task_graph.mark_failed(node.id, str(e))
            raise
    
    async def execute_nodes_parallel(
        self,
        nodes: List[TaskNode],
        task_graph: TaskGraph,
        context: Dict[str, Any],
    ) -> List[Any]:
        """
        Execute multiple nodes in parallel.
        
        Args:
            nodes: List of ready nodes
            task_graph: Parent task graph
            context: Execution context
            
        Returns:
            List of results
        """
        tasks = [
            self.execute_node(node, task_graph, context)
            for node in nodes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {nodes[i].id} failed: {result}")
        
        return results
    
    async def execute_graph(
        self,
        task_graph: TaskGraph,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete task graph.
        
        Iteratively executes ready nodes until all complete
        or a stopping condition is met.
        
        Args:
            task_graph: Task graph to execute
            initial_context: Optional initial context
            
        Returns:
            Final execution context with all results
        """
        context = initial_context or {
            "results": {},
            "sources": [],
            "findings": [],
            "validated_findings": [],
            "claims": [],
        }
        
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check stopping conditions
            should_stop, reason = task_graph.should_stop()
            if should_stop:
                context["stop_reason"] = reason
                break
            
            # Get ready nodes
            ready = task_graph.get_ready_nodes()
            
            if not ready:
                # All done or blocked
                break
            
            # Execute ready nodes in parallel
            await self.execute_nodes_parallel(ready, task_graph, context)
            
            # Check for reflexion result that may require replanning
            for node in ready:
                if node.type == TaskType.REFLEXION and node.status == TaskStatus.COMPLETE:
                    context["reflexion_result"] = node.result
        
        # Final stats
        context["execution_stats"] = {
            "iterations": iteration,
            "total_nodes": len(task_graph.nodes),
            "completed": len(task_graph.get_completed_nodes()),
            "failed": len(task_graph.get_failed_nodes()),
            "total_time_ms": task_graph.metrics["total_time_ms"],
        }
        
        return context
    
    def _update_context(
        self,
        node: TaskNode,
        result: Any,
        context: Dict[str, Any],
    ) -> None:
        """
        Update execution context with task result.
        
        Args:
            node: Completed task node
            result: Task result
            context: Context to update
        """
        # Store result by node ID
        if "results" not in context:
            context["results"] = {}
        context["results"][node.id] = result
        
        # Extract and aggregate based on task type
        if result is None:
            return
        
        content = result.content if hasattr(result, "content") else result
        
        if node.type in [TaskType.SEARCH_WEB, TaskType.SEARCH_ACADEMIC, TaskType.SEARCH_TECHNICAL]:
            # Store search results as findings
            if isinstance(content, dict):
                findings = content.get("findings", content.get("results", []))
            elif isinstance(content, list):
                findings = content
            else:
                findings = []
            
            context.setdefault("sources", []).extend(findings)
            context.setdefault("findings", []).extend(findings)
        
        elif node.type == TaskType.VALIDATE_CLAIMS:
            # Store validated findings
            if isinstance(content, dict):
                validated = content.get("validated_findings", content.get("findings", []))
            elif isinstance(content, list):
                validated = content
            else:
                validated = []
            
            context.setdefault("validated_findings", []).extend(validated)
        
        elif node.type == TaskType.EXTRACT_CLAIMS:
            # Store claims
            if isinstance(content, dict):
                claims = content.get("claims", content.get("claim_objects", []))
            elif isinstance(content, list):
                claims = content
            else:
                claims = []
            
            context.setdefault("claims", []).extend(claims)
    
    def _merge_evidence(
        self,
        node: TaskNode,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge evidence from multiple sources.
        
        Internal operation that doesn't need an agent.
        """
        findings = context.get("validated_findings", [])
        claims = context.get("claims", [])
        
        return {
            "merged_findings": findings,
            "merged_claims": claims,
            "finding_count": len(findings),
            "claim_count": len(claims),
        }
    
    def _deduplicate_claims(
        self,
        node: TaskNode,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deduplicate claims based on similarity.
        
        Simple text-based deduplication for now.
        """
        claims = context.get("claims", [])
        
        seen_texts = set()
        unique_claims = []
        
        for claim in claims:
            text = claim.get("normalized_text", claim.get("claim", "")).lower()
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_claims.append(claim)
        
        return {
            "original_count": len(claims),
            "unique_count": len(unique_claims),
            "deduplicated_claims": unique_claims,
        }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
