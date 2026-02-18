"""
Reflexion Agent for Deep Research Agent

Implements the self-correction loop based on the Reflexion methodology.
Critiques intermediate outputs and generates improved queries when gaps are detected.
"""

import os
import json
from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResult

# V2 imports for planner integration
try:
    from ..planning.task_graph import TaskGraph
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    TaskGraph = None


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts')
    prompt_path = os.path.join(prompts_dir, f'{prompt_name}.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"System prompt not found: {prompt_path}"


class ReflexionAgent(BaseAgent):
    """
    Reflexion Agent - Self-correction and iterative improvement.
    
    Implements the generate → critique → refine loop:
    1. Evaluate intermediate research outputs
    2. Identify knowledge gaps and quality issues
    3. Generate refined queries for follow-up research
    4. Determine when research is complete
    
    V2 Enhancement: Triggers replanning in hierarchical planner when significant gaps detected.
    """
    
    MAX_ITERATIONS = 5  # Maximum reflexion loops to prevent infinite loops
    
    # Thresholds for replanning decision
    REPLAN_GAP_THRESHOLD = 2      # Number of gaps that trigger replan
    REPLAN_QUALITY_THRESHOLD = 0.5  # Quality below this triggers replan
    
    def __init__(
        self,
        llm_client,
        context_manager=None,
        planner=None,
    ):
        """
        Initialize the Reflexion Agent.
        
        Args:
            llm_client: LLM client for critique
            context_manager: Optional context manager
            planner: V2 HierarchicalPlannerAgent for dynamic replanning
        """
        super().__init__(llm_client, context_manager)
        self._system_prompt = None
        self.planner = planner
    
    @property
    def system_prompt(self) -> str:
        """Lazy load system prompt from file."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt('reflexion_prompt')
        return self._system_prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt
    
    def critique(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        current_iteration: int = 0,
    ) -> Dict[str, Any]:
        """
        Critique the current research findings.
        
        Args:
            query: Original research query
            findings: Current validated findings
            current_iteration: Current iteration number
            
        Returns:
            Critique result with recommendations
        """
        # Prepare findings summary
        findings_summary = ""
        for i, f in enumerate(findings[:10], 1):
            score = f.get("reliability_score", "N/A")
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            findings_summary += f"""
{i}. Source: {f.get('source', 'Unknown')}
   Reliability: {score_str}
   Content: {str(f.get('content', ''))[:300]}...
"""
        
        critique_prompt = f"""Critically evaluate the following research findings for the query.

ORIGINAL QUERY: {query}

CURRENT ITERATION: {current_iteration} of {self.MAX_ITERATIONS}

FINDINGS:
{findings_summary}

Evaluate and respond with JSON:
{{
    "overall_quality": 0.0-1.0,
    "query_coverage": "complete|partial|minimal|none",
    "source_diversity": "high|medium|low",
    "key_strengths": ["strength 1", "strength 2"],
    "knowledge_gaps": ["gap 1", "gap 2"],
    "quality_issues": ["issue 1", "issue 2"],
    "continue_research": true/false,
    "stop_reason": "reason if stopping",
    "refined_queries": ["new query if continuing"],
    "priority_focus": "what to focus on next",
    "confidence_in_answer": 0.0-1.0
}}

Be practical: if we're at iteration {current_iteration}, consider whether additional research would significantly improve the answer.
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": critique_prompt}],
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Force stop if max iterations reached
            if current_iteration >= self.MAX_ITERATIONS:
                result["continue_research"] = False
                result["stop_reason"] = "Maximum iterations reached"
            
            # V2: Check if replanning is needed
            result["replan_required"] = self._should_replan(result, current_iteration)
            
            return result
            
        except json.JSONDecodeError:
            # Fallback: assume adequate if we have validated findings
            validated_count = len([f for f in findings if f.get("reliability_score", 0) >= 0.7])
            return {
                "overall_quality": 0.6 if validated_count > 2 else 0.3,
                "query_coverage": "partial",
                "continue_research": validated_count < 3 and current_iteration < self.MAX_ITERATIONS,
                "replan_required": validated_count < 2,  # V2: Replan if very few validated
                "knowledge_gaps": ["Unable to parse critique"],
                "refined_queries": [query],  # Retry original
            }
    
    def _should_replan(
        self,
        critique: Dict[str, Any],
        current_iteration: int,
    ) -> bool:
        """
        V2: Determine if the task graph needs restructuring.
        
        Triggers replanning when:
        - Multiple significant knowledge gaps exist
        - Quality is below threshold
        - Both gaps AND quality issues present
        
        Args:
            critique: The critique result
            current_iteration: Current iteration number
            
        Returns:
            True if replanning is recommended
        """
        # Don't replan on final iteration
        if current_iteration >= self.MAX_ITERATIONS - 1:
            return False
        
        knowledge_gaps = critique.get("knowledge_gaps", [])
        quality_issues = critique.get("quality_issues", [])
        overall_quality = critique.get("overall_quality", 0.5)
        query_coverage = critique.get("query_coverage", "partial")
        
        # Significant gaps present
        has_significant_gaps = len(knowledge_gaps) >= self.REPLAN_GAP_THRESHOLD
        
        # Quality is below threshold
        low_quality = overall_quality < self.REPLAN_QUALITY_THRESHOLD
        
        # Coverage is minimal or none
        poor_coverage = query_coverage in ["minimal", "none"]
        
        # Has both gaps and quality issues
        has_multiple_issues = len(knowledge_gaps) > 0 and len(quality_issues) > 0
        
        return has_significant_gaps or low_quality or poor_coverage or has_multiple_issues
    
    def generate_refined_queries(
        self,
        original_query: str,
        knowledge_gaps: List[str],
        previous_queries: List[str],
    ) -> List[str]:
        """
        Generate refined queries to fill knowledge gaps.
        
        Args:
            original_query: The original research query
            knowledge_gaps: Identified gaps from critique
            previous_queries: Queries already executed
            
        Returns:
            List of new queries to execute
        """
        if not knowledge_gaps:
            return []
        
        refine_prompt = f"""Generate specific search queries to fill the following knowledge gaps.

ORIGINAL QUERY: {original_query}

KNOWLEDGE GAPS:
{json.dumps(knowledge_gaps, indent=2)}

PREVIOUS QUERIES (avoid duplicates):
{json.dumps(previous_queries, indent=2)}

Generate 2-3 NEW, SPECIFIC queries that would fill the gaps.
Respond with JSON array: ["query 1", "query 2", ...]
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": refine_prompt}],
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except:
            # Fallback: turn gaps into queries
            return [f"Research: {gap}" for gap in knowledge_gaps[:2]]
    
    def should_continue(
        self,
        critique: Dict[str, Any],
        current_iteration: int,
    ) -> bool:
        """
        Determine if research should continue.
        
        Args:
            critique: Critique result
            current_iteration: Current iteration number
            
        Returns:
            True if should continue, False if complete
        """
        if current_iteration >= self.MAX_ITERATIONS:
            return False
        
        # Check explicit continue flag
        if not critique.get("continue_research", True):
            return False
        
        # Check quality thresholds
        quality = critique.get("overall_quality", 0)
        confidence = critique.get("confidence_in_answer", 0)
        
        if quality >= 0.8 and confidence >= 0.8:
            return False  # High quality, can stop
        
        # Check if there are actionable gaps
        gaps = critique.get("knowledge_gaps", [])
        if not gaps:
            return False  # No gaps to fill
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the reflexion agent.
        
        Args:
            input_data: Must contain 'query', 'findings', 'iteration'
            
        Returns:
            AgentResult with critique and recommendations
        """
        query = input_data.get("query")
        findings = input_data.get("findings", [])
        iteration = input_data.get("iteration", 0)
        previous_queries = input_data.get("previous_queries", [])
        mode = input_data.get("mode", "deep")
        
        if not query:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error="No query provided",
            )
        
        try:
            if mode == "quick":
                self.log("Quick mode enabled; skipping reflexion")
                return AgentResult(
                    success=True,
                    content={
                        "overall_quality": 0.0,
                        "query_coverage": "partial",
                        "continue_research": False,
                        "replan_required": False,
                        "reflexion_triggered": False,
                        "knowledge_gaps": [],
                        "quality_issues": [],
                        "stop_reason": "quick mode",
                        "refined_queries": [],
                    },
                    agent_name=self.name,
                    metadata={
                        "iteration": iteration,
                        "continue_research": False,
                        "num_refined_queries": 0,
                        "quality_score": 0.0,
                    },
                )

            # Generate critique
            self.log(f"Critiquing iteration {iteration}")
            critique = self.critique(query, findings, iteration)
            critique["reflexion_triggered"] = False
            
            # Determine if we should continue
            should_continue = self.should_continue(critique, iteration)
            
            # Generate refined queries if continuing
            refined_queries = []
            if should_continue:
                gaps = critique.get("knowledge_gaps", [])
                refined_queries = self.generate_refined_queries(
                    query, gaps, previous_queries
                )
                critique["refined_queries"] = refined_queries
            
            # Update context if available
            if self.context and self.context.current_session_id:
                for gap in critique.get("knowledge_gaps", []):
                    self.context.add_knowledge_gap(gap)
                self.context.increment_iteration()
            
            return AgentResult(
                success=True,
                content=critique,
                agent_name=self.name,
                metadata={
                    "iteration": iteration,
                    "continue_research": should_continue,
                    "num_refined_queries": len(refined_queries),
                    "quality_score": critique.get("overall_quality", 0),
                },
            )
        
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error=str(e),
            )
