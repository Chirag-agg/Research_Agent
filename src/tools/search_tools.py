"""
Search Tools for Deep Research Agent

Unified interface for various search APIs used by Specialized Search Agents.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Standardized search result"""
    title: str
    url: str
    content: str
    score: float
    source_api: str


class SearchTools:
    """
    Unified search interface for multiple APIs.
    
    Supported providers:
    - Exa: AI-native neural search
    - Tavily: Agentic web search
    - Firecrawl: Web scraping and extraction
    """
    
    def __init__(self):
        self.exa_key = os.getenv("EXA_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    
    def exa_search(
        self,
        query: str,
        num_results: int = 5,
        search_type: str = "neural",
    ) -> List[SearchResult]:
        """
        Search using Exa API.
        
        Args:
            query: Search query
            num_results: Number of results
            search_type: "neural" or "keyword"
            
        Returns:
            List of SearchResult objects
        """
        if not self.exa_key:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.exa_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
            "useAutoprompt": True,
            "contents": {"text": True},
        }
        
        try:
            response = requests.post(
                "https://api.exa.ai/search",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            return [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("text", "")[:2000],
                    score=item.get("score", 0),
                    source_api="exa",
                )
                for item in data.get("results", [])
            ]
        except Exception as e:
            print(f"Exa search error: {e}")
            return []
    
    def tavily_search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "advanced",
    ) -> List[SearchResult]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            num_results: Number of results
            search_depth: "basic" or "advanced"
            
        Returns:
            List of SearchResult objects
        """
        if not self.tavily_key:
            return []
        
        payload = {
            "api_key": self.tavily_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": num_results,
            "include_raw_content": True,
        }
        
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            data = response.json()
            return [
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:2000],
                    score=item.get("score", 0),
                    source_api="tavily",
                )
                for item in data.get("results", [])
            ]
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    def firecrawl_scrape(
        self,
        url: str,
        formats: List[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape a URL using Firecrawl API.
        
        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, etc.)
            
        Returns:
            Scraped content dict or None
        """
        if not self.firecrawl_key:
            return None
        
        if formats is None:
            formats = ["markdown"]
        
        headers = {
            "Authorization": f"Bearer {self.firecrawl_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "url": url,
            "formats": formats,
        }
        
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                "url": url,
                "content": data.get("data", {}).get("markdown", ""),
                "metadata": data.get("data", {}).get("metadata", {}),
            }
        except Exception as e:
            print(f"Firecrawl scrape error: {e}")
            return None
    
    def search(
        self,
        query: str,
        provider: str = "auto",
        num_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search using specified or auto-selected provider.
        
        Args:
            query: Search query
            provider: "exa", "tavily", or "auto"
            num_results: Number of results
            
        Returns:
            List of SearchResult objects
        """
        if provider == "exa" or (provider == "auto" and self.exa_key):
            return self.exa_search(query, num_results)
        elif provider == "tavily" or (provider == "auto" and self.tavily_key):
            return self.tavily_search(query, num_results)
        else:
            return []
