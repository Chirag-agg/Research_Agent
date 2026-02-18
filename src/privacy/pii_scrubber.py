"""
Deep Research Agent V2 - PII Scrubber

Privacy protection for research content.
Detects and removes Personally Identifiable Information (PII).
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Represents a detected PII match."""
    pii_type: str
    value: str
    start: int
    end: int
    replacement: str


class PIIScrubber:
    """
    PII Scrubber - Detect and remove personally identifiable information.
    
    Supports:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Names (with optional LLM assistance)
    """
    
    # Regex patterns for PII detection
    PATTERNS = {
        "email": {
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "replacement": "[EMAIL REDACTED]",
        },
        "phone_us": {
            "pattern": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
            "replacement": "[PHONE REDACTED]",
        },
        "phone_intl": {
            "pattern": r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            "replacement": "[PHONE REDACTED]",
        },
        "ssn": {
            "pattern": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
            "replacement": "[SSN REDACTED]",
        },
        "credit_card": {
            "pattern": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
            "replacement": "[CARD REDACTED]",
        },
        "ip_address": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "replacement": "[IP REDACTED]",
        },
        "ipv6": {
            "pattern": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
            "replacement": "[IP REDACTED]",
        },
        "date_of_birth": {
            "pattern": r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b",
            "replacement": "[DOB REDACTED]",
        },
        "passport": {
            "pattern": r"\b[A-Z]{1,2}\d{6,9}\b",
            "replacement": "[PASSPORT REDACTED]",
        },
    }
    
    # Country-specific patterns
    COUNTRY_PATTERNS = {
        "uk_nino": {
            "pattern": r"\b[A-Z]{2}\d{6}[A-Z]\b",
            "replacement": "[NINO REDACTED]",
        },
        "uk_postcode": {
            "pattern": r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
            "replacement": "[POSTCODE REDACTED]",
        },
    }
    
    def __init__(
        self,
        llm_client=None,
        include_country_patterns: bool = True,
        custom_patterns: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize the PII Scrubber.
        
        Args:
            llm_client: Optional LLM client for enhanced detection
            include_country_patterns: Include country-specific patterns
            custom_patterns: Additional custom patterns
        """
        self.llm_client = llm_client
        
        # Combine patterns
        self.patterns = dict(self.PATTERNS)
        if include_country_patterns:
            self.patterns.update(self.COUNTRY_PATTERNS)
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Compile regex patterns
        self._compiled = {
            name: re.compile(config["pattern"], re.IGNORECASE)
            for name, config in self.patterns.items()
        }
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            List of PIIMatch objects
        """
        matches = []
        
        for pii_type, pattern in self._compiled.items():
            replacement = self.patterns[pii_type]["replacement"]
            
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    replacement=replacement,
                ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start)
        
        return matches
    
    def scrub(self, text: str) -> str:
        """
        Remove detected PII from text.
        
        Args:
            text: Text to scrub
            
        Returns:
            Text with PII replaced
        """
        matches = self.detect(text)
        
        if not matches:
            return text
        
        # Build result by replacing from end to start
        # (to preserve position indices)
        result = text
        for match in reversed(matches):
            result = result[:match.start] + match.replacement + result[match.end:]
        
        return result
    
    def scrub_with_report(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scrub PII and return a report of what was removed.
        
        Args:
            text: Text to scrub
            
        Returns:
            Tuple of (scrubbed_text, report)
        """
        matches = self.detect(text)
        
        report = {
            "total_matches": len(matches),
            "by_type": {},
            "matches": [],
        }
        
        for match in matches:
            # Count by type
            report["by_type"][match.pii_type] = report["by_type"].get(match.pii_type, 0) + 1
            # Store match info (but not the actual value for privacy)
            report["matches"].append({
                "type": match.pii_type,
                "position": (match.start, match.end),
            })
        
        scrubbed = self.scrub(text)
        
        return scrubbed, report
    
    async def scrub_with_llm(self, text: str) -> str:
        """
        Use LLM for enhanced PII detection.
        
        Catches contextual PII that regex might miss,
        such as names, addresses, and implicit identifiers.
        
        Args:
            text: Text to scrub
            
        Returns:
            Text with PII replaced
        """
        if not self.llm_client:
            return self.scrub(text)
        
        # First pass with regex
        text = self.scrub(text)
        
        # Second pass with LLM for contextual detection
        prompt = f"""Identify any remaining personally identifiable information (PII) in this text.

Look for:
- Full names of real people (not researchers being cited)
- Street addresses
- Medical record numbers
- Bank account numbers
- Device identifiers
- Biometric identifiers

TEXT:
{text[:3000]}

Return JSON with replacements:
{{"replacements": [{{"original": "...", "replacement": "[TYPE REDACTED]"}}]}}

If no PII found, return: {{"replacements": []}}"""

        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": "You detect PII for privacy protection. Be thorough but don't over-redact public information like researcher names in citations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            
            import json
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # Apply LLM-detected replacements
            for rep in result.get("replacements", []):
                original = rep.get("original", "")
                replacement = rep.get("replacement", "[REDACTED]")
                if original:
                    text = text.replace(original, replacement)
            
            return text
            
        except Exception as e:
            # Fall back to regex-only scrubbing
            return text
    
    def add_pattern(
        self,
        name: str,
        pattern: str,
        replacement: str,
    ) -> None:
        """
        Add a custom pattern.
        
        Args:
            name: Pattern name
            pattern: Regex pattern
            replacement: Replacement text
        """
        self.patterns[name] = {
            "pattern": pattern,
            "replacement": replacement,
        }
        self._compiled[name] = re.compile(pattern, re.IGNORECASE)
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported PII types."""
        return list(self.patterns.keys())


# Convenience functions
def scrub_pii(text: str) -> str:
    """Quick utility to scrub PII from text."""
    scrubber = PIIScrubber()
    return scrubber.scrub(text)


def detect_pii(text: str) -> List[Dict[str, Any]]:
    """Quick utility to detect PII in text."""
    scrubber = PIIScrubber()
    matches = scrubber.detect(text)
    return [
        {"type": m.pii_type, "position": (m.start, m.end)}
        for m in matches
    ]
