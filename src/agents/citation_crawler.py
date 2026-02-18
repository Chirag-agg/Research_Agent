"""
Deep Research Agent V2 - Citation Crawler Agent

Implements recursive citation following for multi-hop research.
Follows citation chains to find primary sources and deeper evidence.
"""

import re
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseAgent, AgentResult

# V2 imports
try:
    from ..memory.models import Source
    from ..evidence.graph import EvidenceGraph
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
    Source = None
    EvidenceGraph = None


@dataclass
class Citation:
    """Represents a citation extracted from a source."""
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source_type: str = "unknown"  # paper, website, book, etc.
    relevance_score: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "source_type": self.source_type,
            "relevance_score": self.relevance_score,
        }


class CitationCrawlerAgent(BaseAgent):
    """
    Citation Crawler Agent - Recursive citation following.
    
    Capabilities:
    1. Extract citations from source content
    2. Prioritize citations by relevance
    3. Recursively follow citation chains
    4. Track citation depth and prevent cycles
    """
    
    # Limits to prevent runaway crawling
    MAX_DEPTH = 3
    MAX_CITATIONS_PER_SOURCE = 10
    MAX_TOTAL_CITATIONS = 50
    
    def __init__(
        self,
        llm_client,
        context_manager=None,
        search_agent=None,
        memory_api=None,
        evidence_graph: Optional["EvidenceGraph"] = None,
    ):
        """
        Initialize the Citation Crawler.
        
        Args:
            llm_client: LLM client for extraction
            context_manager: Optional context manager
            search_agent: WebSearchAgent for fetching cited sources
            memory_api: V2 memory API for storage
            evidence_graph: V2 evidence graph for linking
        """
        super().__init__(llm_client)
        self.context_manager = context_manager
        self.search_agent = search_agent
        self.memory_api = memory_api
        self.evidence_graph = evidence_graph
        
        # Track visited URLs to prevent cycles
        self._visited_urls: Set[str] = set()
        self._citation_count = 0
    
    def extract_citations(
        self,
        content: str,
        source_url: Optional[str] = None,
    ) -> List[Citation]:
        """
        Extract citations from source content using LLM.
        
        Args:
            content: Text content to extract from
            source_url: Source URL for context
            
        Returns:
            List of Citation objects
        """
        # Truncate content if too long
        max_content = 6000
        if len(content) > max_content:
            content = content[:max_content] + "..."
        
        prompt = f"""Extract citations and references from this content.

CONTENT:
{content}

For each citation, extract:
- title: Full title of the cited work
- authors: List of author names
- year: Publication year (if available)
- doi: DOI if present
- url: Direct URL if present
- source_type: paper | website | book | report | other
- relevance_score: 0-1 (how central to the argument)

Return JSON array, max {self.MAX_CITATIONS_PER_SOURCE} most relevant citations:
[{{"title": "...", "authors": [...], "year": 2024, ...}}]

JSON citations:"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You extract academic citations from text. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            
            content = response.content.strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            citations_data = json.loads(content)
            
            # Convert to Citation objects
            citations = []
            for c in citations_data[:self.MAX_CITATIONS_PER_SOURCE]:
                if isinstance(c, dict) and c.get("title"):
                    citations.append(Citation(
                        title=c.get("title", ""),
                        authors=c.get("authors", []),
                        year=c.get("year"),
                        doi=c.get("doi"),
                        url=c.get("url"),
                        source_type=c.get("source_type", "unknown"),
                        relevance_score=min(1.0, max(0.0, float(c.get("relevance_score", 0.5)))),
                    ))
            
            return citations
            
        except Exception as e:
            self.log(f"Citation extraction failed: {e}")
            return []
    
    def extract_citations_heuristic(self, content: str) -> List[Citation]:
        """
        Fallback heuristic citation extraction using patterns.
        
        Args:
            content: Text to extract from
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Pattern for DOI
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        dois = re.findall(doi_pattern, content)
        for doi in dois[:5]:
            citations.append(Citation(
                title=f"DOI: {doi}",
                doi=doi,
                url=f"https://doi.org/{doi}",
                source_type="paper",
                relevance_score=0.7,
            ))
        
        # Pattern for arXiv
        arxiv_pattern = r'arXiv:(\d{4}\.\d{4,})'
        arxiv_ids = re.findall(arxiv_pattern, content)
        for arxiv_id in arxiv_ids[:5]:
            citations.append(Citation(
                title=f"arXiv: {arxiv_id}",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                source_type="paper",
                relevance_score=0.8,
            ))
        
        # Pattern for URLs in references section
        url_pattern = r'https?://[^\s<>"\']+(?:\.pdf|/abstract|/paper)'
        urls = re.findall(url_pattern, content)
        for url in urls[:5]:
            if url not in [c.url for c in citations]:
                citations.append(Citation(
                    title=url.split("/")[-1][:50],
                    url=url,
                    source_type="paper" if ".pdf" in url else "website",
                    relevance_score=0.5,
                ))
        
        return citations[:self.MAX_CITATIONS_PER_SOURCE]
    
    def prioritize_citations(
        self,
        citations: List[Citation],
        query: str,
    ) -> List[Citation]:
        """
        Prioritize citations by relevance to the research query.
        
        Args:
            citations: List of extracted citations
            query: Research query for context
            
        Returns:
            Sorted list of citations
        """
        # Simple scoring - could be enhanced with embeddings
        query_terms = set(query.lower().split())
        
        for citation in citations:
            # Boost score if title contains query terms
            title_terms = set(citation.title.lower().split())
            overlap = len(query_terms.intersection(title_terms))
            if overlap > 0:
                citation.relevance_score = min(1.0, citation.relevance_score + 0.1 * overlap)
            
            # Boost academic sources
            if citation.source_type == "paper":
                citation.relevance_score = min(1.0, citation.relevance_score + 0.1)
            
            # Boost if has DOI
            if citation.doi:
                citation.relevance_score = min(1.0, citation.relevance_score + 0.1)
        
        # Sort by relevance
        return sorted(citations, key=lambda c: -c.relevance_score)
    
    async def crawl_citation(
        self,
        citation: Citation,
        current_depth: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch and process a single citation.
        
        Args:
            citation: Citation to crawl
            current_depth: Current crawl depth
            
        Returns:
            Fetched content or None
        """
        if not self.search_agent:
            return None
        
        # Build search query from citation
        search_query = citation.title
        if citation.authors:
            search_query += f" {citation.authors[0]}"
        if citation.year:
            search_query += f" {citation.year}"
        
        # Use search agent to find the source
        try:
            result = self.search_agent.execute({
                "query": search_query,
                "num_results": 1,
            })
            
            if result.success and result.content:
                findings = result.content.get("findings", [])
                if findings:
                    finding = findings[0]
                    finding["citation_depth"] = current_depth
                    finding["cited_by"] = citation.to_dict()
                    return finding
        except Exception as e:
            self.log(f"Failed to fetch citation: {e}")
        
        return None
    
    async def crawl_citations_recursive(
        self,
        source_content: str,
        source_url: str,
        query: str,
        current_depth: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Recursively crawl citations with depth limiting.
        
        Args:
            source_content: Content to extract citations from
            source_url: URL of the source
            query: Research query for prioritization
            current_depth: Current crawl depth
            
        Returns:
            List of crawled sources with citation metadata
        """
        if current_depth >= self.MAX_DEPTH:
            return []
        
        if self._citation_count >= self.MAX_TOTAL_CITATIONS:
            return []
        
        if source_url in self._visited_urls:
            return []
        
        self._visited_urls.add(source_url)
        
        # Extract citations
        citations = self.extract_citations(source_content, source_url)
        if not citations:
            citations = self.extract_citations_heuristic(source_content)
        
        # Prioritize
        citations = self.prioritize_citations(citations, query)
        
        # Crawl top citations
        crawled_sources = []
        for citation in citations[:5]:  # Limit per level
            if self._citation_count >= self.MAX_TOTAL_CITATIONS:
                break
            
            url = citation.url or ""
            if url and url in self._visited_urls:
                continue
            
            source = await self.crawl_citation(citation, current_depth + 1)
            if source:
                self._citation_count += 1
                crawled_sources.append(source)
                
                # Recursively crawl if not at max depth
                if current_depth + 1 < self.MAX_DEPTH:
                    nested = await self.crawl_citations_recursive(
                        source.get("content", ""),
                        source.get("url", source.get("source", "")),
                        query,
                        current_depth + 1,
                    )
                    crawled_sources.extend(nested)
        
        return crawled_sources
    
    def reset_crawler(self):
        """Reset crawler state for new session."""
        self._visited_urls.clear()
        self._citation_count = 0
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the citation crawler.
        
        Args:
            input_data: Must contain 'sources' or 'content', and 'query'
            
        Returns:
            AgentResult with crawled citations
        """
        sources = input_data.get("sources", [])
        content = input_data.get("content", "")
        query = input_data.get("query", "")
        max_depth = input_data.get("max_depth", self.MAX_DEPTH)
        
        self.MAX_DEPTH = min(max_depth, 5)  # Safety cap
        self.reset_crawler()
        
        try:
            all_citations = []
            
            if sources:
                # Extract from multiple sources
                for source in sources:
                    src_content = source.get("content", source.get("text", ""))
                    src_url = source.get("url", source.get("source", ""))
                    
                    citations = self.extract_citations(src_content, src_url)
                    if not citations:
                        citations = self.extract_citations_heuristic(src_content)
                    
                    citations = self.prioritize_citations(citations, query)
                    all_citations.extend([c.to_dict() for c in citations])
                    
            elif content:
                # Single content extraction
                citations = self.extract_citations(content)
                if not citations:
                    citations = self.extract_citations_heuristic(content)
                
                citations = self.prioritize_citations(citations, query)
                all_citations = [c.to_dict() for c in citations]
            
            # Deduplicate by title
            seen_titles = set()
            unique_citations = []
            for c in all_citations:
                title_key = c.get("title", "").lower()[:50]
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_citations.append(c)
            
            return AgentResult(
                success=True,
                content={
                    "citations": unique_citations[:self.MAX_TOTAL_CITATIONS],
                    "count": len(unique_citations),
                },
                agent_name="CitationCrawlerAgent",
                metadata={
                    "sources_processed": len(sources) if sources else 1,
                    "max_depth": self.MAX_DEPTH,
                },
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                content=None,
                agent_name="CitationCrawlerAgent",
                error=str(e),
            )
