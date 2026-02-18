"""
Deep Research Agent V2 - Claim Extractor Agent

Extracts factual claims from source content for the evidence graph.
Uses LLM to identify, normalize, and score claims.
"""

import json
import os
from typing import Dict, Any, List, Optional

from .base import BaseAgent, AgentResult
from ..memory.models import Claim, EvidenceRelation


def load_prompt(prompt_name: str) -> str:
    """Load a system prompt from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")
    prompt_path = os.path.join(prompts_dir, prompt_name)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


class ClaimExtractorAgent(BaseAgent):
    """
    Claim Extractor Agent - Extract factual claims from source content.
    
    Responsibilities:
    1. Parse source content for factual statements
    2. Normalize claims to canonical form
    3. Assign confidence scores
    4. Link claims to supporting text
    """
    
    def __init__(self, llm_client, context_manager=None, memory_api=None):
        """
        Initialize the Claim Extractor.
        
        Args:
            llm_client: LLM client for extraction
            context_manager: Optional context manager
            memory_api: Optional memory API for claim storage
        """
        super().__init__(llm_client)
        self.context_manager = context_manager
        self.memory_api = memory_api
        self._system_prompt = None
    
    @property
    def system_prompt(self) -> str:
        """Lazy load system prompt from file."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt("claim_extraction_prompt.md")
            if not self._system_prompt:
                self._system_prompt = self._default_prompt()
        return self._system_prompt
    
    def _default_prompt(self) -> str:
        """Return default prompt if file not found."""
        return """You are a precise claim extractor.

Input: article text

Output: JSON list of factual claims with:
- claim: short, canonicalized factual statement
- supporting_text: exact sentence(s) from the article
- confidence: 0-1 based on clarity and verifiability

Rules:
- Extract only factual claims, not opinions
- Canonicalize: remove hedging, use present tense
- Do not hallucinate claims not in the text
- Limit to 10 most important claims

Example output:
[
  {
    "claim": "Transformer architectures use self-attention mechanisms",
    "supporting_text": "The transformer, introduced in 2017, relies primarily on self-attention...",
    "confidence": 0.95
  }
]"""
    
    def get_system_prompt(self) -> str:
        return self.system_prompt
    
    def extract_claims(
        self,
        content: str,
        source_url: Optional[str] = None,
        max_claims: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Extract factual claims from text content.
        
        Args:
            content: Text content to extract from
            source_url: Optional source URL for context
            max_claims: Maximum claims to extract
            
        Returns:
            List of claim dictionaries
        """
        # Truncate content if too long
        max_content = 8000
        if len(content) > max_content:
            content = content[:max_content] + "..."
        
        prompt = f"""Extract factual claims from this content:

{content}

Return as JSON array with at most {max_claims} claims.
Each claim must have: claim, supporting_text, confidence

JSON claims:"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low for consistency
                max_tokens=2000,
            )
            
            content = response.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            claims = json.loads(content)
            
            # Validate and normalize claims
            normalized = []
            for claim in claims[:max_claims]:
                if isinstance(claim, dict) and "claim" in claim:
                    normalized.append({
                        "claim": claim.get("claim", ""),
                        "supporting_text": claim.get("supporting_text", ""),
                        "confidence": min(1.0, max(0.0, float(claim.get("confidence", 0.5)))),
                        "source_url": source_url,
                    })
            
            return normalized
            
        except Exception as e:
            self.log(f"Claim extraction failed: {e}")
            return []
    
    def normalize_claim(self, claim_text: str) -> str:
        """
        Normalize a claim to canonical form for deduplication.
        
        - Remove hedging words
        - Use present tense
        - Remove filler words
        - Lowercase
        
        Args:
            claim_text: Original claim text
            
        Returns:
            Normalized claim text
        """
        # Simple normalization - can be enhanced with LLM
        normalized = claim_text.lower().strip()
        
        # Remove common hedging
        hedges = [
            "it is believed that", "research suggests that",
            "studies show that", "according to experts",
            "it appears that", "evidence indicates that",
            "it is thought that", "many believe that",
        ]
        
        for hedge in hedges:
            normalized = normalized.replace(hedge, "")
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        return normalized
    
    def extract_claims_batch(
        self,
        sources: List[Dict[str, Any]],
        max_claims_per_source: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from multiple sources.
        
        Args:
            sources: List of source dictionaries with 'content' key
            max_claims_per_source: Max claims per source
            
        Returns:
            List of all extracted claims with source references
        """
        all_claims = []
        
        for source in sources:
            content = source.get("content", source.get("text", ""))
            source_url = source.get("url", source.get("source", ""))
            source_id = source.get("id", source.get("source_id", ""))
            
            if not content:
                continue
            
            claims = self.extract_claims(
                content=content,
                source_url=source_url,
                max_claims=max_claims_per_source,
            )
            
            # Add source reference to each claim
            for claim in claims:
                claim["source_id"] = source_id
                claim["source_url"] = source_url
                claim["normalized_text"] = self.normalize_claim(claim["claim"])
            
            all_claims.extend(claims)
        
        return all_claims
    
    def create_claim_objects(
        self,
        claim_dicts: List[Dict[str, Any]],
    ) -> List[Claim]:
        """
        Convert claim dictionaries to Claim objects.
        
        Args:
            claim_dicts: List of claim dictionaries
            
        Returns:
            List of Claim objects
        """
        claims = []
        
        for c in claim_dicts:
            claim = Claim(
                text=c.get("claim", ""),
                normalized_text=c.get("normalized_text", self.normalize_claim(c.get("claim", ""))),
                confidence=c.get("confidence", 0.5),
                supporting_text=c.get("supporting_text", ""),
                provenance=[c.get("source_id", "")] if c.get("source_id") else [],
            )
            claims.append(claim)
        
        return claims
    
    async def extract_and_store(
        self,
        sources: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> List[str]:
        """
        Extract claims and store them via MemoryAPI.
        
        Args:
            sources: List of source dictionaries
            session_id: Optional session ID
            
        Returns:
            List of stored claim IDs
        """
        if not self.memory_api:
            self.log("No MemoryAPI available, returning claims without storage")
            claims = self.extract_claims_batch(sources)
            return [c.get("claim", "") for c in claims]
        
        claim_ids = []
        
        for source in sources:
            content = source.get("content", source.get("text", ""))
            source_id = source.get("id", source.get("source_id", ""))
            
            if not content or not source_id:
                continue
            
            # Extract claims
            claims = self.extract_claims(content=content, source_url=source.get("url", ""))
            
            # Store each claim
            for claim_dict in claims:
                claim = Claim(
                    text=claim_dict["claim"],
                    normalized_text=self.normalize_claim(claim_dict["claim"]),
                    confidence=claim_dict["confidence"],
                    supporting_text=claim_dict["supporting_text"],
                )
                
                # Determine relation based on confidence
                relation = EvidenceRelation.SUPPORTS if claim_dict["confidence"] > 0.6 else EvidenceRelation.MENTIONS
                
                claim_id = await self.memory_api.add_claim(
                    claim=claim,
                    source_id=source_id,
                    relation=relation,
                    strength=claim_dict["confidence"],
                )
                claim_ids.append(claim_id)
        
        return claim_ids
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the claim extractor.
        
        Args:
            input_data: Must contain 'sources' (list) or 'content' (string)
            
        Returns:
            AgentResult with extracted claims
        """
        sources = input_data.get("sources", [])
        content = input_data.get("content", "")
        source_url = input_data.get("source_url", input_data.get("url", ""))
        max_claims = input_data.get("max_claims", 10)
        
        try:
            if sources:
                # Batch extraction
                claims = self.extract_claims_batch(
                    sources=sources,
                    max_claims_per_source=max_claims // len(sources) if sources else max_claims,
                )
            elif content:
                # Single content extraction
                claims = self.extract_claims(
                    content=content,
                    source_url=source_url,
                    max_claims=max_claims,
                )
            else:
                return AgentResult(
                    success=False,
                    content=None,
                    agent_name="ClaimExtractorAgent",
                    error="Either 'sources' or 'content' is required",
                )
            
            # Create Claim objects
            claim_objects = self.create_claim_objects(claims)
            
            return AgentResult(
                success=True,
                content={
                    "claims": claims,
                    "claim_objects": [c.to_dict() for c in claim_objects],
                    "count": len(claims),
                },
                agent_name="ClaimExtractorAgent",
                metadata={
                    "source_count": len(sources) if sources else 1,
                    "claims_extracted": len(claims),
                },
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name="ClaimExtractorAgent",
                error=str(e),
            )
