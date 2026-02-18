"""
Deep Research Agent V2 - Task Graph

Data structures for hierarchical task planning:
- TaskNode: Individual task in the graph
- TaskGraph: DAG of tasks with dependencies
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum


class TaskType(Enum):
    """Types of tasks in the research workflow."""
    # Search tasks
    SEARCH_WEB = "search.web"
    SEARCH_ACADEMIC = "search.academic"
    SEARCH_TECHNICAL = "search.technical"
    SEARCH_CITATION = "search.citation_recursive"
    
    # Processing tasks
    EXTRACT_CLAIMS = "extract_claims"
    VALIDATE_CLAIMS = "validate_claims"
    MERGE_EVIDENCE = "merge_evidence"
    DEDUPLICATE_CLAIMS = "deduplicate_claims"
    
    # Synthesis tasks
    SYNTHESIZE_SECTION = "synthesize_section"
    SYNTHESIZE_REPORT = "synthesize_report"
    
    # Control tasks
    REFLEXION = "reflexion"
    REPLAN = "replan"


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    READY = "ready"      # Dependencies satisfied, can execute
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModelTier(Enum):
    """Model size tiers for routing."""
    SMALL = "small"      # Fast, cheap - sanitization, claim extraction
    MEDIUM = "medium"    # Balanced - validation, citation parsing
    LARGE = "large"      # High quality - final synthesis


@dataclass
class TaskNode:
    """
    A single task in the task graph.
    
    Represents an atomic unit of work with:
    - Type and input parameters
    - Dependencies on other tasks
    - Budget constraints
    - Model routing hints
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType = TaskType.SEARCH_WEB
    input: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    budget_ms: int = 5000                # Time budget in milliseconds
    stop_criteria: Dict[str, Any] = field(default_factory=dict)
    preferred_agent: str = ""            # Agent class name
    model_hint: ModelTier = ModelTier.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        result_val = self.result
        
        # Handle non-serializable result types
        if result_val is not None:
             if hasattr(result_val, "__dataclass_fields__"):
                 from dataclasses import asdict
                 result_val = asdict(result_val)
             elif hasattr(result_val, "to_dict"):
                 result_val = result_val.to_dict()
             # Basic types are fine, but complex objects need conversion
             elif not isinstance(result_val, (str, int, float, bool, list, dict, type(None))):
                 result_val = str(result_val)
                 
        return {
            "id": self.id,
            "type": self.type.value,
            "input": self.input,
            "dependencies": self.dependencies,
            "budget_ms": self.budget_ms,
            "stop_criteria": self.stop_criteria,
            "preferred_agent": self.preferred_agent,
            "model_hint": self.model_hint.value,
            "status": self.status.value,
            "result": result_val,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskNode":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=TaskType(data.get("type", "search.web")),
            input=data.get("input", {}),
            dependencies=data.get("dependencies", []),
            budget_ms=data.get("budget_ms", 5000),
            stop_criteria=data.get("stop_criteria", {}),
            preferred_agent=data.get("preferred_agent", ""),
            model_hint=ModelTier(data.get("model_hint", "medium")),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 2),
        )
    
    def is_search_task(self) -> bool:
        """Check if this is a search task."""
        return self.type.value.startswith("search.")
    
    def is_synthesis_task(self) -> bool:
        """Check if this is a synthesis task."""
        return self.type.value.startswith("synthesize")
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class GoalCriteria:
    """
    Criteria for determining when research goals are met.
    """
    coverage: float = 0.95           # Target query coverage
    confidence: float = 0.8          # Target confidence score
    max_iterations: int = 3          # Maximum reflexion iterations
    min_sources: int = 5             # Minimum number of sources
    min_claims: int = 3              # Minimum number of validated claims
    max_time_ms: int = 180000        # Maximum total time (3 minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage,
            "confidence": self.confidence,
            "max_iterations": self.max_iterations,
            "min_sources": self.min_sources,
            "min_claims": self.min_claims,
            "max_time_ms": self.max_time_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalCriteria":
        return cls(
            coverage=data.get("coverage", 0.95),
            confidence=data.get("confidence", 0.8),
            max_iterations=data.get("max_iterations", 3),
            min_sources=data.get("min_sources", 5),
            min_claims=data.get("min_claims", 3),
            max_time_ms=data.get("max_time_ms", 180000),
        )


class TaskGraph:
    """
    Directed Acyclic Graph of research tasks.
    
    Manages task dependencies, execution order, and goal tracking.
    Supports dynamic replanning based on reflexion feedback.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize an empty task graph.
        
        Args:
            session_id: Optional session ID for persistence
        """
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.nodes: Dict[str, TaskNode] = {}
        self.goal_criteria = GoalCriteria()
        self.current_iteration = 0
        self.metrics: Dict[str, Any] = {
            "coverage": 0.0,
            "confidence": 0.0,
            "sources_count": 0,
            "claims_count": 0,
            "total_time_ms": 0,
        }
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_node(self, node: TaskNode) -> str:
        """
        Add a task node to the graph.
        
        Args:
            node: TaskNode to add
            
        Returns:
            Node ID
        """
        self.nodes[node.id] = node
        self._update_node_status(node.id)
        self.updated_at = datetime.now()
        return node.id
    
    def get_node(self, node_id: str) -> Optional[TaskNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.
        
        Also removes this node from any dependency lists.
        """
        if node_id not in self.nodes:
            return False
        
        del self.nodes[node_id]
        
        # Remove from dependencies
        for node in self.nodes.values():
            if node_id in node.dependencies:
                node.dependencies.remove(node_id)
        
        self.updated_at = datetime.now()
        return True
    
    def add_dependency(self, from_id: str, to_id: str) -> bool:
        """
        Add a dependency: from_id depends on to_id.
        
        Args:
            from_id: Task that depends on another
            to_id: Task that must complete first
            
        Returns:
            True if added successfully
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        
        if to_id not in self.nodes[from_id].dependencies:
            self.nodes[from_id].dependencies.append(to_id)
            self._update_node_status(from_id)
        
        return True
    
    def get_ready_nodes(self) -> List[TaskNode]:
        """
        Get nodes ready for execution.
        
        A node is ready if:
        - Status is PENDING or READY
        - All dependencies are COMPLETE
        
        Returns:
            List of ready TaskNodes
        """
        ready = []
        
        for node in self.nodes.values():
            if node.status in [TaskStatus.PENDING, TaskStatus.READY]:
                deps_complete = all(
                    self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETE
                    for dep_id in node.dependencies
                )
                if deps_complete:
                    node.status = TaskStatus.READY
                    ready.append(node)
        
        return ready
    
    def get_pending_nodes(self) -> List[TaskNode]:
        """Get all pending nodes."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.PENDING]
    
    def get_running_nodes(self) -> List[TaskNode]:
        """Get all currently running nodes."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.RUNNING]
    
    def get_completed_nodes(self) -> List[TaskNode]:
        """Get all completed nodes."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.COMPLETE]
    
    def get_failed_nodes(self) -> List[TaskNode]:
        """Get all failed nodes."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]
    
    def mark_running(self, node_id: str) -> bool:
        """Mark a node as running."""
        if node_id not in self.nodes:
            return False
        
        self.nodes[node_id].status = TaskStatus.RUNNING
        self.nodes[node_id].started_at = datetime.now()
        return True
    
    def mark_complete(self, node_id: str, result: Any = None) -> bool:
        """
        Mark a node as complete with its result.
        
        Args:
            node_id: Task ID
            result: Task result
            
        Returns:
            True if updated
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.status = TaskStatus.COMPLETE
        node.result = result
        node.completed_at = datetime.now()
        
        # Calculate duration
        if node.started_at:
            duration = (node.completed_at - node.started_at).total_seconds() * 1000
            self.metrics["total_time_ms"] += duration
        
        # Update dependent nodes' status
        for other_id, other_node in self.nodes.items():
            if node_id in other_node.dependencies:
                self._update_node_status(other_id)
        
        self.updated_at = datetime.now()
        return True
    
    def mark_failed(self, node_id: str, error: str) -> bool:
        """
        Mark a node as failed.
        
        Args:
            node_id: Task ID
            error: Error message
            
        Returns:
            True if updated
        """
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.error = error
        node.completed_at = datetime.now()
        
        # Check if can retry
        if node.can_retry():
            node.retry_count += 1
            node.status = TaskStatus.PENDING
        else:
            node.status = TaskStatus.FAILED
        
        self.updated_at = datetime.now()
        return True
    
    def _update_node_status(self, node_id: str) -> None:
        """Update a node's status based on dependencies."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        if node.status in [TaskStatus.RUNNING, TaskStatus.COMPLETE, 
                          TaskStatus.FAILED, TaskStatus.SKIPPED]:
            return
        
        deps_complete = all(
            self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETE
            for dep_id in node.dependencies
        )
        
        if deps_complete:
            node.status = TaskStatus.READY
        else:
            node.status = TaskStatus.PENDING
    
    def set_goal_criteria(self, **kwargs) -> None:
        """
        Update goal criteria.
        
        Args:
            **kwargs: Any GoalCriteria fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.goal_criteria, key):
                setattr(self.goal_criteria, key, value)
    
    def update_metrics(
        self,
        coverage: Optional[float] = None,
        confidence: Optional[float] = None,
        sources_count: Optional[int] = None,
        claims_count: Optional[int] = None,
    ) -> None:
        """Update current metrics."""
        if coverage is not None:
            self.metrics["coverage"] = coverage
        if confidence is not None:
            self.metrics["confidence"] = confidence
        if sources_count is not None:
            self.metrics["sources_count"] = sources_count
        if claims_count is not None:
            self.metrics["claims_count"] = claims_count
    
    def goal_met(self) -> bool:
        """
        Check if the research goals are met.
        
        Returns:
            True if all goal criteria satisfied
        """
        criteria = self.goal_criteria
        metrics = self.metrics
        
        return (
            metrics["coverage"] >= criteria.coverage and
            metrics["confidence"] >= criteria.confidence and
            metrics["sources_count"] >= criteria.min_sources and
            metrics["claims_count"] >= criteria.min_claims
        )
    
    def should_stop(self) -> Tuple[bool, str]:
        """
        Check if research should stop.
        
        Returns:
            (should_stop, reason)
        """
        criteria = self.goal_criteria
        
        # Goal met
        if self.goal_met():
            return True, "Goals achieved"
        
        # Max iterations
        if self.current_iteration >= criteria.max_iterations:
            return True, f"Max iterations ({criteria.max_iterations}) reached"
        
        # Time limit
        if self.metrics["total_time_ms"] >= criteria.max_time_ms:
            return True, f"Time limit ({criteria.max_time_ms}ms) exceeded"
        
        # All tasks complete
        incomplete = [n for n in self.nodes.values() 
                     if n.status not in [TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.SKIPPED]]
        if not incomplete:
            return True, "All tasks complete"
        
        return False, ""
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get tasks in execution order (levels of parallelism).
        
        Returns:
            List of lists, where each inner list contains
            task IDs that can run in parallel
        """
        levels = []
        completed: Set[str] = set()
        
        while len(completed) < len(self.nodes):
            # Find nodes with all deps in completed
            level = []
            for node_id, node in self.nodes.items():
                if node_id in completed:
                    continue
                if all(dep in completed for dep in node.dependencies):
                    level.append(node_id)
            
            if not level:
                # Cycle detected or remaining nodes have unmet deps
                remaining = [nid for nid in self.nodes if nid not in completed]
                levels.append(remaining)
                break
            
            levels.append(level)
            completed.update(level)
        
        return levels
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize task graph to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "goal_criteria": self.goal_criteria.to_dict(),
            "current_iteration": self.current_iteration,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """Deserialize task graph from dictionary."""
        graph = cls(session_id=data.get("session_id"))
        graph.id = data.get("id", graph.id)
        graph.goal_criteria = GoalCriteria.from_dict(data.get("goal_criteria", {}))
        graph.current_iteration = data.get("current_iteration", 0)
        graph.metrics = data.get("metrics", graph.metrics)
        graph.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        graph.updated_at = datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now()
        
        for node_data in data.get("nodes", {}).values():
            graph.nodes[node_data["id"]] = TaskNode.from_dict(node_data)
        
        return graph
    
    def __repr__(self) -> str:
        status_counts = {}
        for node in self.nodes.values():
            status_counts[node.status.value] = status_counts.get(node.status.value, 0) + 1
        
        return f"TaskGraph(nodes={len(self.nodes)}, status={status_counts}, iteration={self.current_iteration})"

