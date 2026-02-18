"""
Master Planning Agent (MPA) for Deep Research Agent

The central nervous system of the DRA. Responsible for:
- Adaptive query decomposition
- Strategic delegation to Specialized Search Agents
- Synthesis of findings into coherent reports
- Context steering for long-horizon tasks
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .base import BaseAgent, AgentResult

# V2 imports for evidence graph integration
try:
    from ..evidence.graph import EvidenceGraph
    from ..memory.models import Claim
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    EvidenceGraph = None
    Claim = None


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts')
    prompt_path = os.path.join(prompts_dir, f'{prompt_name}.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"System prompt not found: {prompt_path}"


class MasterPlannerAgent(BaseAgent):
    """
    Master Planning Agent - Orchestrates the entire research process.
    
    Takes a user query and:
    1. Decomposes it into sub-queries for parallel execution
    2. Delegates to specialized search agents
    3. Synthesizes results into a final report
    4. Steers context based on reflexion feedback
    
    V2 Enhancement: Claim-based synthesis with evidence graph support.
    """
    
    def __init__(
        self,
        llm_client,
        context_manager=None,
        evidence_graph: Optional["EvidenceGraph"] = None,
    ):
        """
        Initialize the Master Planner Agent.
        
        Args:
            llm_client: LLM client for inference
            context_manager: Optional context manager
            evidence_graph: V2 evidence graph for claim provenance
        """
        super().__init__(llm_client, context_manager)
        self._system_prompt = None
        self.evidence_graph = evidence_graph
    
    @property
    def system_prompt(self) -> str:
        """Lazy load system prompt from file."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt('master_planner_prompt')
        return self._system_prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """
        Decompose a complex query into sub-queries for parallel search.
        
        Args:
            query: The main research query
            
        Returns:
            Dictionary with sub-queries and agent assignments
        """
        decomposition_prompt = f"""Decompose the following research query into sub-queries for parallel search.

QUERY: {query}

Respond with a JSON object in this exact format:
{{
    "main_query": "the original query",
    "research_type": "general|academic|technical|mixed",
    "sub_queries": [
        {{
            "query": "specific sub-query 1",
            "agent": "web_search|academic|technical",
            "priority": 1,
            "rationale": "why this sub-query is needed"
        }},
        ...
    ],
    "synthesis_strategy": "how to combine the results",
    "expected_output_format": "what the final answer should look like"
}}

Generate 3-5 sub-queries that together comprehensively answer the main query.
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": decomposition_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.3,  # Lower temperature for structured output
        )
        
        try:
            # Extract JSON from response
            content = response.content
            # Try to find JSON in the response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Store sub-queries in context if available
            if self.context and self.context.current_session_id:
                session = self.context.get_session()
                session.sub_queries = [sq["query"] for sq in result.get("sub_queries", [])]
            
            return result
            
        except json.JSONDecodeError as e:
            self.log(f"Failed to parse decomposition JSON: {e}")
            # Return a simple fallback structure
            return {
                "main_query": query,
                "research_type": "general",
                "sub_queries": [
                    {"query": query, "agent": "web_search", "priority": 1, "rationale": "Direct search"}
                ],
                "synthesis_strategy": "direct",
                "raw_response": response.content,
            }
    
    def synthesize_findings(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        format_type: str = "markdown",
    ) -> str:
        """
        Synthesize findings from multiple agents into a coherent report.
        
        Args:
            query: The original research query
            findings: List of findings from search agents
            format_type: Output format (markdown, json, plain)
            
        Returns:
            Synthesized report string
        """
        findings_text = ""
        total_length = 0
        MAX_CONTEXT = 30000 # Safe limit for Gemma 3 (~8k tokens)

        for i, finding in enumerate(findings, 1):
            if total_length > MAX_CONTEXT:
                findings_text += f"\n... [{len(findings) - i + 1} more findings truncated due to length limit] ..."
                break
                
            score = finding.get("reliability_score", "N/A")
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            
            # Truncate individual content to balance coverage
            content = finding.get('content', 'No content')
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
                
            entry = f"""
### Source {i}
- **Agent**: {finding.get('agent', 'Unknown')}
- **Source**: {finding.get('source', 'Unknown')}
- **Reliability**: {score_str}
- **Content**: {content}
"""
            findings_text += entry
            total_length += len(entry)
        
        synthesis_prompt = f"""Based on the following research findings, create a comprehensive synthesis that answers the original query.

ORIGINAL QUERY: {query}

FINDINGS:
{findings_text}

Create a well-structured {format_type} report that:
1. Directly answers the main query
2. Synthesizes information from multiple sources
3. Cites sources clearly with reliability indicators
4. Highlights any gaps or limitations in the research
5. Provides a confidence assessment

Format the output as clean {format_type}.
STRICT OUTPUT RULES:
- Start DIRECTLY with the report title (e.g., # Title).
- Do NOT output "Here is the report" or "Alright, let's...".
- Do NOT output the research plan, JSON blocks, or sub-queries.
- Do NOT output the "Thought Process".
- Output ONLY the final report content.
- Ensure the report is COMPLETE and not truncated. Prioritize finishing the report over extreme detail if needed.
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.5,
            max_tokens=8192,
        )
        
        return response.content
    
    # V2: Enhanced synthesis methods
    
    def cluster_claims(
        self,
        claims: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        V2: Cluster claims by topic for organized synthesis.
        
        Args:
            claims: List of claim dictionaries
            
        Returns:
            List of topic clusters with claims
        """
        if not claims:
            return []
        
        # Use LLM to cluster claims by topic
        # Use LLM to cluster claims by topic
        # Truncate claims to fit context
        claims_text = ""
        for i, c in enumerate(claims[:50]): # Limit to top 50 claims
            text = c.get('claim', c.get('text', ''))
            if len(claims_text) + len(text) > 20000:
                break
            claims_text += f"{i+1}. {text}\n"
        
        prompt = f"""Group these claims into 3-5 topic clusters.

CLAIMS:
{claims_text}

Return JSON:
{{
    "clusters": [
        {{
            "topic": "topic name",
            "summary": "one sentence summary",
            "claim_indices": [1, 2, 5]
        }}
    ]
}}
"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            result = json.loads(content)
            
            # Build clusters with actual claims
            clusters = []
            for cluster_data in result.get("clusters", []):
                cluster_claims = []
                for idx in cluster_data.get("claim_indices", []):
                    if 0 < idx <= len(claims):
                        cluster_claims.append(claims[idx - 1])
                
                clusters.append({
                    "topic": cluster_data.get("topic", "Unknown"),
                    "summary": cluster_data.get("summary", ""),
                    "claims": cluster_claims,
                })
            
            return clusters
            
        except Exception as e:
            self.log(f"Claim clustering failed: {e}")
            # Fallback: single cluster
            return [{
                "topic": "Research Findings",
                "summary": "All claims from the research",
                "claims": claims,
            }]
    
    def synthesize_with_claims(
        self,
        query: str,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        V2: Synthesize using claim clusters with evidence annotations.
        
        Returns both markdown and JSON outputs.
        
        Args:
            query: Original research query
            claims: Extracted claims with provenance
            sources: Validated sources
            
        Returns:
            Dictionary with 'markdown' and 'json' keys
        """
        # Cluster claims by topic
        clusters = self.cluster_claims(claims)
        
        # Annotate claims with evidence if graph available
        if self.evidence_graph and HAS_V2:
            for cluster in clusters:
                for claim in cluster.get("claims", []):
                    claim_id = claim.get("id", "")
                    if claim_id:
                        provenance = self.evidence_graph.get_claim_provenance(claim_id)
                        if provenance:
                            claim["supporting_sources"] = provenance.get("supporting_sources", [])
                            claim["contradictions"] = provenance.get("contradicting_sources", [])
                            claim["confidence"] = provenance.get("confidence", claim.get("confidence", 0.5))
        
        # Build synthesis prompt with clusters
        clusters_text = ""
        for i, cluster in enumerate(clusters, 1):
            clusters_text += f"\n### Topic {i}: {cluster['topic']}\n"
            clusters_text += f"Summary: {cluster['summary']}\n\n"
            for j, claim in enumerate(cluster.get('claims', [])[:5], 1):
                conf = claim.get('confidence', 0.5)
                clusters_text += f"{j}. {claim.get('claim', claim.get('text', ''))} [confidence: {conf:.2f}]\n"
        
        synthesis_prompt = f"""Synthesize these research findings into a comprehensive report.

QUERY: {query}

CLUSTERED FINDINGS:
{clusters_text}

Create a report that:
1. Organizes findings by topic
2. Indicates confidence for each major claim
3. Notes any contradictions
4. Includes a confidence summary at the end

Format as clean markdown.
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": synthesis_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.4,
            max_tokens=8192,
        )
        
        markdown_report = response.content
        
        # Build JSON output
        json_output = {
            "query": query,
            "generated_at": datetime.now().isoformat(),
            "clusters": [
                {
                    "topic": c["topic"],
                    "summary": c["summary"],
                    "claims": [
                        {
                            "text": claim.get("claim", claim.get("text", "")),
                            "confidence": claim.get("confidence", 0.5),
                            "sources": claim.get("supporting_sources", claim.get("provenance", [])),
                        }
                        for claim in c.get("claims", [])
                    ],
                }
                for c in clusters
            ],
            "sources": [
                {
                    "url": s.get("url", s.get("source", "")),
                    "title": s.get("title", ""),
                    "reliability": s.get("reliability_score", 0.5),
                }
                for s in sources
            ],
            "metadata": {
                "total_claims": sum(len(c.get("claims", [])) for c in clusters),
                "total_sources": len(sources),
                "topic_count": len(clusters),
            },
        }
        
        return {
            "markdown": markdown_report,
            "json": json_output,
        }
    
    def identify_gaps(
        self,
        query: str,
        current_findings: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify knowledge gaps in the current research findings.
        
        Args:
            query: The original research query
            current_findings: Current findings to evaluate
            
        Returns:
            List of identified knowledge gaps
        """
        findings_summary = "\n".join([
            f"- {f.get('content', 'No content')[:200]}..." 
            for f in current_findings[:5]
        ])
        
        gap_prompt = f"""Analyze the following research findings and identify any knowledge gaps or missing information needed to fully answer the query.

QUERY: {query}

CURRENT FINDINGS SUMMARY:
{findings_summary}

Identify specific knowledge gaps. Respond with a JSON array of gaps:
["gap 1 description", "gap 2 description", ...]

Only include actual gaps. If the research is comprehensive, respond with an empty array: []
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": gap_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.3,
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            gaps = json.loads(content.strip())
            
            # Store in context if available
            if self.context and self.context.current_session_id:
                for gap in gaps:
                    self.context.add_knowledge_gap(gap)
            
            return gaps
            
        except json.JSONDecodeError:
            return []
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the master planning agent.
        
        Args:
            input_data: Must contain 'query' key, optionally 'action' key
            
        Returns:
            AgentResult with planning/synthesis results
        """
        query = input_data.get("query")
        if not query:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error="No query provided",
            )
        
        action = input_data.get("action", "decompose")
        
        try:
            if action == "decompose":
                # Create session if context manager available
                if self.context:
                    self.context.create_session(query)
                
                result = self.decompose_query(query)
                return AgentResult(
                    success=True,
                    content=result,
                    agent_name=self.name,
                    metadata={"action": "decompose"},
                )
            
            elif action == "synthesize":
                findings = input_data.get("findings", [])
                result = self.synthesize_findings(query, findings)
                return AgentResult(
                    success=True,
                    content=result,
                    agent_name=self.name,
                    metadata={"action": "synthesize"},
                )
            
            elif action == "identify_gaps":
                findings = input_data.get("findings", [])
                result = self.identify_gaps(query, findings)
                return AgentResult(
                    success=True,
                    content=result,
                    agent_name=self.name,
                    metadata={"action": "identify_gaps"},
                )
            
            else:
                return AgentResult(
                    success=False,
                    content=None,
                    agent_name=self.name,
                    error=f"Unknown action: {action}",
                )
        
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error=str(e),
            )
