"""
Deep Research Agent V2 - Hierarchical Planner Agent

Replaces flat query decomposition with dynamic task graph planning.
Generates a DAG of research tasks with dependencies, budgets, and model hints.
"""

import json
import os
from typing import Dict, Any, List, Optional

from .base import BaseAgent, AgentResult
from ..planning.task_graph import (
    TaskGraph, TaskNode, TaskType, TaskStatus, ModelTier, GoalCriteria
)


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")
    prompt_path = os.path.join(prompts_dir, prompt_name)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


class HierarchicalPlannerAgent(BaseAgent):
    """
    Hierarchical Planner Agent - V2 Planning System
    
    Generates task graphs for research workflows:
    1. Decomposes query into high-level goals
    2. Expands goals into concrete task nodes
    3. Assigns dependencies, budgets, and model hints
    4. Supports dynamic replanning based on reflexion feedback
    """
    
    def __init__(self, llm_client, context_manager=None, memory_api=None):
        """
        Initialize the Hierarchical Planner.
        
        Args:
            llm_client: LLM client for planning
            context_manager: Optional context manager
            memory_api: Optional memory API for claim retrieval
        """
        super().__init__(llm_client)
        self.context_manager = context_manager
        self.memory_api = memory_api
        self._system_prompt = None
    
    @property
    def system_prompt(self) -> str:
        """Lazy load system prompt from file."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt("hierarchical_planner_prompt.md")
        return self._system_prompt
    
    def get_system_prompt(self) -> str:
        return self.system_prompt
    
    def hierarchical_plan(
        self,
        query: str,
        preferences: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        previous_summary: Optional[str] = None,
        previous_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> TaskGraph:
        """
        Generate a complete task graph for a research query.
        
        Args:
            query: Research query to plan
            preferences: Optional user preferences (depth, time_budget, etc.)
            session_id: Optional session ID for linking
            
        Returns:
            TaskGraph with all nodes and dependencies
        """
        prefs = preferences or {}
        
        # Create task graph
        graph = TaskGraph(session_id=session_id)
        
        mode = prefs.get("mode", "deep")
        quick_mode = mode == "quick"
        # Set goal criteria from preferences
        graph.set_goal_criteria(
            coverage=prefs.get("coverage", 0.95 if quick_mode else 0.97),
            confidence=prefs.get("confidence", 0.7 if quick_mode else 0.85),
            max_iterations=prefs.get("max_iterations", 1 if quick_mode else 3),
            min_sources=prefs.get("min_sources", 3 if quick_mode else 6),
            max_time_ms=prefs.get("max_time_ms", 60000 if quick_mode else 300000),
        )
        
        # Track task IDs for dependency linking
        search_task_ids = []
        extract_task_ids = []
        goals = self._decompose_into_goals(query, prefs, previous_summary, previous_messages)
        if quick_mode:
            goals = goals[:1]  # single layer focus
        
        # Expand each goal into tasks
        top_k = prefs.get("top_k", 5)
        for goal in goals:
            tasks = self._expand_goal(goal, query, top_k=top_k)
            
            for task in tasks:
                graph.add_node(task)
                
                if task.is_search_task():
                    search_task_ids.append(task.id)
        
        # Add claim extraction tasks (depend on searches)
        extract_task = TaskNode(
            type=TaskType.EXTRACT_CLAIMS,
            input={"query": query},
            dependencies=search_task_ids,
            budget_ms=5000,
            model_hint=ModelTier.SMALL,
            preferred_agent="ClaimExtractorAgent",
        )
        graph.add_node(extract_task)
        extract_task_ids.append(extract_task.id)
        
        # Add validation task (depends on extraction) unless in quick mode
        merge_dependencies = [extract_task.id]
        if not quick_mode:
            validate_task = TaskNode(
                type=TaskType.VALIDATE_CLAIMS,
                input={"query": query},
                dependencies=extract_task_ids,
                budget_ms=8000,
                model_hint=ModelTier.MEDIUM,
                preferred_agent="SourceValidatorAgent",
            )
            graph.add_node(validate_task)
            merge_dependencies = [validate_task.id]
        
        # Add evidence merge task
        merge_task = TaskNode(
            type=TaskType.MERGE_EVIDENCE,
            input={},
            dependencies=merge_dependencies,
            budget_ms=3000,
            model_hint=ModelTier.SMALL,
        ) 
        graph.add_node(merge_task)
        
        # Add reflexion task unless in quick mode
        synth_dependencies = [merge_task.id]
        if not quick_mode:
            reflexion_task = TaskNode(
                type=TaskType.REFLEXION,
                input={"query": query},
                dependencies=[merge_task.id],
                budget_ms=5000,
                model_hint=ModelTier.MEDIUM,
                preferred_agent="ReflexionAgent",
            )
            graph.add_node(reflexion_task)
            synth_dependencies = [reflexion_task.id]
        
        # Add synthesis task
        synth_task = TaskNode(
            type=TaskType.SYNTHESIZE_REPORT,
            input={"query": query},
            dependencies=synth_dependencies,
            budget_ms=12000 if quick_mode else 15000,
            model_hint=ModelTier.MEDIUM if quick_mode else ModelTier.LARGE,
            preferred_agent="MasterPlannerAgent",
        )
        graph.add_node(synth_task)
        
        return graph
    
    def _decompose_into_goals(
        self,
        query: str,
        preferences: Dict[str, Any],
        previous_summary: Optional[str] = None,
        previous_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decompose query into high-level research goals.
        
        Uses LLM to identify distinct aspects of the query
        that need investigation.
        
        Returns:
            List of goal dictionaries with focus areas
        """
        context_snippets = []
        if previous_summary:
                context_snippets.append(f"Previous summary:\n{previous_summary}")
        if previous_messages:
                last_msgs = previous_messages[-5:]
                msgs_text = "\n".join(
                        f"- {m.get('role','user')}: {m.get('content','')}" for m in last_msgs
                )
                context_snippets.append(f"Recent conversation:\n{msgs_text}")
        prior_context = "\n\n".join(context_snippets) if context_snippets else "None"

        prompt = f"""Analyze this research query and identify 3-5 distinct research goals.

Query: {query}

Prior context (use to refine and avoid restarting work):
{prior_context}

For each goal, specify:
1. focus: The specific aspect to investigate
2. search_types: Which search types are most relevant (web, academic, technical)
3. priority: 1 (critical) to 3 (supplementary)
4. depth: shallow (quick overview) or deep (thorough investigation)

Return as JSON array. Example:
[
    {{"focus": "Current state of technology", "search_types": ["web", "technical"], "priority": 1, "depth": "deep"}},
    {{"focus": "Academic research and papers", "search_types": ["academic"], "priority": 1, "depth": "deep"}},
    {{"focus": "Industry adoption patterns", "search_types": ["web"], "priority": 2, "depth": "shallow"}}
]

JSON goals:"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            
            # Parse response
            content = response.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            goals = json.loads(content)
            return goals
            
        except Exception as e:
            self.log(f"Goal decomposition failed: {e}, using defaults")
            
            # Default goals
            return [
                {
                    "focus": "Current information and developments",
                    "search_types": ["web"],
                    "priority": 1,
                    "depth": "deep"
                },
                {
                    "focus": "Academic and research perspective",
                    "search_types": ["academic"],
                    "priority": 1,
                    "depth": "deep"
                },
                {
                    "focus": "Technical implementations",
                    "search_types": ["technical"],
                    "priority": 2,
                    "depth": "shallow"
                }
            ]
    
    def _expand_goal(
        self,
        goal: Dict[str, Any],
        original_query: str,
        top_k: int,
    ) -> List[TaskNode]:
        """
        Expand a goal into concrete task nodes.
        
        Args:
            goal: Goal dictionary with focus, search_types, etc.
            original_query: Original research query
            
        Returns:
            List of TaskNodes
        """
        tasks = []
        focus = goal.get("focus", "")
        search_types = goal.get("search_types", ["web"])
        priority = goal.get("priority", 2)
        depth = goal.get("depth", "shallow")
        
        # Calculate budget based on priority and depth
        base_budget = 5000 if depth == "shallow" else 10000
        budget = base_budget // priority
        
        # Create search tasks for each type
        for search_type in search_types:
            task_type = {
                "web": TaskType.SEARCH_WEB,
                "academic": TaskType.SEARCH_ACADEMIC,
                "technical": TaskType.SEARCH_TECHNICAL,
            }.get(search_type, TaskType.SEARCH_WEB)
            
            # Generate focused query
            search_query = f"{original_query} {focus}" if focus else original_query
            
            task = TaskNode(
                type=task_type,
                input={
                    "query": search_query,
                    "focus": focus,
                    "num_results": min(top_k, 8) if depth == "deep" else min(top_k, 5),
                },
                dependencies=[],  # Search tasks have no deps
                budget_ms=budget,
                model_hint=ModelTier.SMALL if depth == "shallow" else ModelTier.MEDIUM,
                preferred_agent={
                    "web": "WebSearchAgent",
                    "academic": "AcademicSearchAgent",
                    "technical": "TechnicalSearchAgent",
                }.get(search_type, "WebSearchAgent"),
            )
            tasks.append(task)
        
        return tasks
    
    def replan(
        self,
        task_graph: TaskGraph,
        reflexion_feedback: Dict[str, Any],
        preferences: Optional[Dict[str, Any]] = None,
    ) -> TaskGraph:
        """
        Update task graph based on reflexion feedback.
        
        Adds new search tasks to fill identified gaps.
        
        Args:
            task_graph: Current task graph
            reflexion_feedback: Feedback from reflexion agent
            
        Returns:
            Updated TaskGraph
        """
        prefs = preferences or {}
        quick_mode = prefs.get("mode") == "quick"

        if quick_mode:
            return task_graph
        if not reflexion_feedback.get("replan_required", False):
            return task_graph
        
        # Increment iteration
        task_graph.current_iteration += 1
        
        # Get refined queries from feedback
        refined_queries = reflexion_feedback.get("refined_queries", [])
        knowledge_gaps = reflexion_feedback.get("knowledge_gaps", [])
        priority_focus = reflexion_feedback.get("priority_focus", "")
        
        # Find the reflexion task to add new dependencies
        reflexion_tasks = [
            n for n in task_graph.nodes.values()
            if n.type == TaskType.REFLEXION and n.status != TaskStatus.COMPLETE
        ]
        
        new_search_ids = []
        
        # Add new search tasks for gaps
        for i, gap in enumerate(knowledge_gaps[:3]):  # Limit to 3 new searches
            # Determine search type based on gap content
            search_type = TaskType.SEARCH_WEB
            if any(term in gap.lower() for term in ["paper", "research", "study", "academic"]):
                search_type = TaskType.SEARCH_ACADEMIC
            elif any(term in gap.lower() for term in ["code", "implementation", "github", "technical"]):
                search_type = TaskType.SEARCH_TECHNICAL
            
            task = TaskNode(
                type=search_type,
                input={
                    "query": refined_queries[i] if i < len(refined_queries) else gap,
                    "focus": gap,
                    "num_results": 3,
                    "iteration": task_graph.current_iteration,
                },
                dependencies=[],
                budget_ms=5000,
                model_hint=ModelTier.MEDIUM,
            )
            task_graph.add_node(task)
            new_search_ids.append(task.id)
        
        # Add new extraction task for new searches
        if new_search_ids:
            extract_task = TaskNode(
                type=TaskType.EXTRACT_CLAIMS,
                input={"iteration": task_graph.current_iteration},
                dependencies=new_search_ids,
                budget_ms=3000,
                model_hint=ModelTier.SMALL,
            )
            task_graph.add_node(extract_task)
            
            # Add new validation task
            validate_task = TaskNode(
                type=TaskType.VALIDATE_CLAIMS,
                input={"iteration": task_graph.current_iteration},
                dependencies=[extract_task.id],
                budget_ms=5000,
                model_hint=ModelTier.MEDIUM,
            )
            task_graph.add_node(validate_task)
            
            # Add new reflexion task
            if not quick_mode:
                new_reflexion = TaskNode(
                    type=TaskType.REFLEXION,
                    input={"iteration": task_graph.current_iteration},
                    dependencies=[validate_task.id],
                    budget_ms=5000,
                    model_hint=ModelTier.MEDIUM,
                )
                task_graph.add_node(new_reflexion)
                
                # Update synthesis to depend on new reflexion
                for node in task_graph.nodes.values():
                    if node.type == TaskType.SYNTHESIZE_REPORT:
                        node.dependencies = [new_reflexion.id]
        
        return task_graph
    
    def plan_from_llm(
        self,
        query: str,
        preferences: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> TaskGraph:
        """
        Use LLM to generate the complete task graph.
        
        More flexible than hierarchical_plan but slower.
        
        Args:
            query: Research query
            preferences: User preferences
            session_id: Session ID
            
        Returns:
            TaskGraph generated by LLM
        """
        prefs = preferences or {}
        
        prompt = f"""{self.system_prompt}

Research Query: {query}

Preferences:
- Max time: {prefs.get('max_time_ms', 180000)}ms
- Depth: {prefs.get('depth', 'balanced')}
- Focus areas: {prefs.get('focus_areas', 'general')}

Generate a task graph as JSON:"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            
            content = response.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # Build graph from LLM output
            graph = TaskGraph(session_id=session_id)
            
            if "goal_criteria" in data:
                graph.goal_criteria = GoalCriteria.from_dict(data["goal_criteria"])
            
            for node_data in data.get("nodes", []):
                # Map string type to enum
                type_str = node_data.get("type", "search.web")
                try:
                    task_type = TaskType(type_str)
                except ValueError:
                    task_type = TaskType.SEARCH_WEB
                
                # Map model hint
                hint_str = node_data.get("model_hint", "medium")
                try:
                    model_hint = ModelTier(hint_str)
                except ValueError:
                    model_hint = ModelTier.MEDIUM
                
                node = TaskNode(
                    id=node_data.get("id", None) or str(uuid.uuid4()),
                    type=task_type,
                    input=node_data.get("input", {}),
                    dependencies=node_data.get("deps", node_data.get("dependencies", [])),
                    budget_ms=node_data.get("budget_ms", 5000),
                    model_hint=model_hint,
                )
                graph.add_node(node)
            
            return graph
            
        except Exception as e:
            self.log(f"LLM planning failed: {e}, using default plan")
            return self.hierarchical_plan(query, preferences, session_id)
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the hierarchical planner.
        
        Args:
            input_data: Must contain 'query', optionally 'action', 'preferences'
            
        Returns:
            AgentResult with task graph
        """
        query = input_data.get("query", "")
        action = input_data.get("action", "plan")
        preferences = input_data.get("preferences", {})
        session_id = input_data.get("session_id")
        
        if not query and action != "replan":
            return AgentResult(
                success=False,
                content=None,
                agent_name="HierarchicalPlannerAgent",
                error="Query is required",
            )
        
        try:
            if action == "plan":
                task_graph = self.hierarchical_plan(query, preferences, session_id)
            elif action == "plan_llm":
                task_graph = self.plan_from_llm(query, preferences, session_id)
            elif action == "replan":
                existing_graph = input_data.get("task_graph")
                feedback = input_data.get("reflexion_feedback", {})
                if not existing_graph:
                    return AgentResult(
                        success=False,
                        content=None,
                        agent_name="HierarchicalPlannerAgent",
                        error="task_graph required for replan",
                    )
                task_graph = self.replan(existing_graph, feedback)
            else:
                return AgentResult(
                    success=False,
                    content=None,
                    agent_name="HierarchicalPlannerAgent",
                    error=f"Unknown action: {action}",
                )
            
            return AgentResult(
                success=True,
                content=task_graph.to_dict(),
                agent_name="HierarchicalPlannerAgent",
                metadata={
                    "node_count": len(task_graph.nodes),
                    "execution_order": task_graph.get_execution_order(),
                },
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name="HierarchicalPlannerAgent",
                error=str(e),
            )


# Import uuid for node ID generation
import uuid
