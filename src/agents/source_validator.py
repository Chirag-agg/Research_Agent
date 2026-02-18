"""
Source Validation Agent (SVA) for Deep Research Agent

The proprietary component that validates source reliability and claim truthfulness.
Uses LLM-as-a-Judge framework with specialized prompting for veracity assessment.
"""

import os
import json
from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResult

# V2 imports for evidence graph integration
try:
    from ..memory.models import EvidenceRelation
    from ..evidence.graph import EvidenceGraph
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    EvidenceGraph = None
    EvidenceRelation = None


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts')
    prompt_path = os.path.join(prompts_dir, f'{prompt_name}.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"System prompt not found: {prompt_path}"


class SourceValidatorAgent(BaseAgent):
    """
    Source Validation Agent - Fact-checking and source reliability scoring.
    
    Evaluates findings for:
    1. Source credibility (domain authority, publication type)
    2. Content veracity (claim accuracy, citation support)
    3. Recency and relevance
    4. Cross-source consistency
    
    V2 Enhancement: Integrates with evidence graph for claim-source linking.
    """
    
    def __init__(
        self,
        llm_client,
        context_manager=None,
        evidence_graph: Optional["EvidenceGraph"] = None,
        memory_api=None,
    ):
        """
        Initialize the Source Validator Agent.
        
        Args:
            llm_client: LLM client for validation
            context_manager: Optional context manager
            evidence_graph: V2 evidence graph for claim-source linking
            memory_api: V2 memory API for claim storage
        """
        super().__init__(llm_client, context_manager)
        self._system_prompt = None
        self.evidence_graph = evidence_graph
        self.memory_api = memory_api
    
    @property
    def system_prompt(self) -> str:
        """Lazy load system prompt from file."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt('source_validator_prompt')
        return self._system_prompt

    def get_system_prompt(self) -> str:
        return self.system_prompt
    
    def validate_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single finding for reliability.
        
        Args:
            finding: Finding dict with 'content', 'source', etc.
            
        Returns:
            Validation result with reliability score and notes
        """
        validation_prompt = f"""Validate the following research finding for reliability and truthfulness.

SOURCE: {finding.get('source', 'Unknown')}
TITLE: {finding.get('title', 'No title')}
CONTENT:
{finding.get('content', 'No content')[:3000]}

Evaluate and respond with JSON:
{{
    "reliability_score": 0.0-1.0,
    "source_credibility": "high|medium|low",
    "content_veracity": "verified|likely|uncertain|questionable",
    "key_claims": ["claim 1", "claim 2"],
    "supported_claims": ["claims with evidence"],
    "unsupported_claims": ["claims without evidence"],
    "potential_issues": ["bias", "outdated", "etc."],
    "validation_notes": "Brief justification for the score",
    "recommendation": "accept|verify|reject"
}}
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": validation_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.2,  # Low temperature for consistent scoring
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # V2: Add evidence edges to graph if available
            if self.evidence_graph and HAS_V2:
                self._add_evidence_edges(finding, result)
            
            return result
            
        except json.JSONDecodeError:
            # Fallback scoring based on source heuristics
            return self._heuristic_validation(finding)
    
    def _add_evidence_edges(
        self,
        finding: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> None:
        """
        V2: Add evidence edges to the evidence graph.
        
        Links validated claims to their source with appropriate relations.
        
        Args:
            finding: The validated finding
            validation: Validation result with claims
        """
        if not self.evidence_graph or not HAS_V2:
            return
        
        source_id = finding.get("source_id", finding.get("id", ""))
        reliability = validation.get("reliability_score", 0.5)
        
        # Process key claims - they get supporting edges
        for claim_text in validation.get("key_claims", []):
            # Add as supporting evidence based on reliability
            relation = (
                EvidenceRelation.SUPPORTS
                if reliability > 0.6
                else EvidenceRelation.MENTIONS
            )
            
            # Find or create claim in graph
            # For now, we use the claim text as a pseudo-ID
            claim_id = claim_text[:100].replace(" ", "_").lower()
            
            self.evidence_graph.add_evidence(
                claim_id=claim_id,
                source_id=source_id,
                relation=relation,
                strength=reliability,
                validation_notes=validation.get("validation_notes", ""),
            )
        
        # Process unsupported claims - they get mentions relation
        for claim_text in validation.get("unsupported_claims", []):
            claim_id = claim_text[:100].replace(" ", "_").lower()
            
            self.evidence_graph.add_evidence(
                claim_id=claim_id,
                source_id=source_id,
                relation=EvidenceRelation.MENTIONS,
                strength=0.3,  # Low strength for unsupported
                validation_notes="Unsupported claim",
            )
    
    def _heuristic_validation(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic-based validation when LLM parsing fails."""
        source = finding.get("source", "").lower()
        
        # Simple heuristics based on domain
        score = 0.5  # Default medium score
        
        if any(d in source for d in [".gov", ".edu", "arxiv.org"]):
            score = 0.85
            credibility = "high"
        elif any(d in source for d in ["wikipedia", "reuters", "bbc", "nature.com"]):
            score = 0.75
            credibility = "high"
        elif any(d in source for d in ["medium.com", "blog", "reddit"]):
            score = 0.55
            credibility = "medium"
        elif "mock" in source or "example.com" in source:
            score = 0.3
            credibility = "low"
        else:
            credibility = "medium"
        
        return {
            "reliability_score": score,
            "source_credibility": credibility,
            "content_veracity": "uncertain",
            "key_claims": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "potential_issues": ["heuristic_only"],
            "validation_notes": "Scored using heuristic fallback",
            "recommendation": "verify" if score < 0.7 else "accept",
        }
    
    def validate_batch(
        self,
        findings: List[Dict[str, Any]],
        use_llm: bool = False,  # Default to heuristics for speed
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of findings.
        
        Args:
            findings: List of findings to validate
            use_llm: If True, use LLM for validation (slower but more accurate)
            
        Returns:
            List of validated findings with scores
        """
        validated = []
        for finding in findings:
            # Use fast heuristic validation by default
            if use_llm:
                validation = self.validate_finding(finding)
            else:
                validation = self._heuristic_validation(finding)
            
            # Merge validation into finding
            validated_finding = {**finding}
            validated_finding["reliability_score"] = validation.get("reliability_score", 0.5)
            validated_finding["validation"] = validation
            validated.append(validated_finding)
            
            # Update context if available
            if self.context and self.context.current_session_id:
                session = self.context.get_session()
                # Find and update the finding in context
                for i, f in enumerate(session.findings):
                    if f.get("source") == finding.get("source"):
                        self.context.mark_source_validated(
                            finding_index=i,
                            reliability_score=validation.get("reliability_score", 0.5),
                            validation_notes=validation.get("validation_notes", ""),
                        )
                        break
        
        return validated
    
    def cross_validate(
        self,
        findings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Cross-validate findings against each other for consistency.
        
        Args:
            findings: Validated findings to cross-check
            
        Returns:
            Cross-validation summary
        """
        if len(findings) < 2:
            return {"consistent": True, "conflicts": []}
        
        # Extract key claims from all findings
        all_claims = []
        for f in findings:
            if "validation" in f:
                all_claims.extend(f["validation"].get("key_claims", []))
        
        cross_check_prompt = f"""Analyze these key claims from different sources for consistency:

CLAIMS:
{json.dumps(all_claims[:20], indent=2)}

Identify any contradictions or significant disagreements between sources.
Respond with JSON:
{{
    "consistent": true/false,
    "conflicts": [
        {{"claim_a": "...", "claim_b": "...", "conflict_type": "contradiction|disagreement|uncertain"}}
    ],
    "consensus_claims": ["claims all sources agree on"],
    "summary": "Brief consistency assessment"
}}
"""
        
        response = self.llm.chat(
            messages=[{"role": "user", "content": cross_check_prompt}],
            system_prompt=self.system_prompt,
            temperature=0.2,
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except:
            return {"consistent": True, "conflicts": [], "summary": "Unable to cross-validate"}
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the source validation agent.
        
        Args:
            input_data: Must contain 'findings' key with list of findings
            
        Returns:
            AgentResult with validated findings
        """
        findings = input_data.get("findings", [])
        if not findings:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error="No findings provided for validation",
            )
        
        try:
            # Validate all findings (using fast heuristics by default)
            self.log(f"Validating {len(findings)} findings")
            validated = self.validate_batch(findings)
            
            # Skip cross-validation for speed (can be enabled later)
            cross_validation = {"consistent": True, "conflicts": [], "summary": "Skipped for performance"}
            
            # Calculate summary stats
            scores = [f.get("reliability_score", 0) for f in validated]
            avg_score = sum(scores) / len(scores) if scores else 0
            accepted = len([s for s in scores if s >= 0.7])
            
            return AgentResult(
                success=True,
                content=validated,
                agent_name=self.name,
                metadata={
                    "num_validated": len(validated),
                    "avg_reliability": round(avg_score, 2),
                    "accepted_count": accepted,
                    "cross_validation": cross_validation,
                },
            )
        
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name=self.name,
                error=str(e),
            )
