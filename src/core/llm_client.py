"""
Together AI & Gemini LLM Client for Deep Research Agent

Provides a unified interface for LLM inference using either Together AI or Google Gemini.
Supports multiple models with automatic fallback and token tracking.
"""

import asyncio
import os
import threading
import time
import requests
from typing import Optional, List, Dict, Any, Generator, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
import httpx

try:
    from together import Together
except ImportError:
    Together = None



try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()


@dataclass
class LLMResponse:
    """Structured response from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """
    Unified LLM Client (OpenRouter, Gemini, Together)
    
    Handles all LLM inference for the Deep Research Agent system.
    Supports streaming, token tracking, and model selection.
    
    V2 Enhancement: Adaptive model routing based on task type and budget.
    """
    
    # Available models
    MODELS = {
        "default": "gemma-3-27b-it",
        "fast": "gemma-3-27b-it",
        "reasoning": "gemma-3-27b-it",
        "gemini-flash": "gemma-3-27b-it",
        "gemini-pro": "gemma-3-27b-it",
    }
    
    # V2: Model tiers for adaptive routing
    # All tiers map to gemma-3-27b-it per user request
    MODEL_TIERS = {
        "small": ["gemma-3-27b-it"],
        "medium": ["gemma-3-27b-it"],
        "large": ["gemma-3-27b-it"],
    }
    
    # V2: Task type to model tier mapping
    # All tasks will eventually route to the single available model in the tiers
    TASK_ROUTING = {
        "sanitize": "small",
        "claim_extract": "small",
        "search_format": "small",
        "validate": "medium",
        "citation_parse": "medium",
        "reflexion": "medium",
        "decomposition": "medium",
        "synthesize": "large",
        "final_report": "large",
    }
    
    
    def __init__(self):
        """Initialize LLM client with auto-discovery of keys."""
        
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.provider = None
        self.client = None
        self.model_usage_breakdown: Dict[str, int] = {}
        
        # V2: Budget tracking per task
        self._task_budgets: Dict[str, int] = {}  # task_id -> remaining_budget_ms
        self._task_token_usage: Dict[str, int] = {}  # task_id -> tokens used
        
        # Checking for model overrides from env vars (e.g., from Modal secrets)
        default_model = os.getenv("DEFAULT_MODEL", "gemma-3-27b-it")
        fast_model = os.getenv("FAST_MODEL", "gemma-3-27b-it") # Use fast model if available or fallback
        
        self.MODELS["default"] = default_model
        self.MODELS["fast"] = fast_model
        self.MODELS["reasoning"] = default_model  # Default to reasoning with the strong model

        # 0. Try Google Gemini (Preferred per user request)
        self.google_key = os.getenv("GEMINI_API_KEY")
        if self.google_key:
            self.provider = "google"
            self.MODELS["default"] = "gemma-3-27b-it"
            print(f"Using Google Gemini provider with key: {self.google_key[:10]}...")
            return

        # 0b. Try Cerebras
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY")
        if self.cerebras_key:
            self.provider = "cerebras"
            # Cerebras supports two chat models; set routing accordingly
            self.MODELS = {
                "default": "gpt-oss-120b",
                "fast": "llama3.1-8b",
                "reasoning": "gpt-oss-120b",
            }
            self.MODEL_TIERS = {
                "small": ["llama3.1-8b"],
                "medium": ["gpt-oss-120b"],
                "large": ["gpt-oss-120b"],
            }
            print("Using Cerebras provider")
            return

        # 1. Try OpenRouter (Secondary)
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if self.openrouter_key:
            if OpenAI is None:
                raise ImportError("openai package required for OpenRouter. pip install openai")
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/YashNuhash/Deep-Research-Agent",
                    "X-Title": "Deep Research Agent"
                }
            )
            self.provider = "openrouter"
            print(f"Using OpenRouter provider with default model: {default_model}")
            return
            
        # 2. Try Together
        self.together_key = os.getenv("TOGETHER_API_KEY")
        if self.together_key and Together:
            self.client = Together(api_key=self.together_key)
            self.provider = "together"
            # Update models for Together if not overridden by env vars with specific OpenRouter names
            if not os.getenv("DEFAULT_MODEL"):
                self.MODELS = {
                    "default": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                    "fast": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    "reasoning": "meta-llama/Llama-3.1-70B-Instruct-Turbo",
                }
            print("Using Together provider")
            return
            
        raise ValueError("No valid API key found. Set GEMINI_API_KEY, OPENROUTER_API_KEY or TOGETHER_API_KEY.")
            
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        """
        start_time = time.time()
        model_name = self.MODELS.get(model, model)
        complexity = self._complexity_from_messages(messages)
        
        if self.provider == "openrouter":
            resp = self._chat_openrouter(messages, model_name, temperature, max_tokens, system_prompt)
        elif self.provider == "together":
            resp = self._chat_together(messages, model_name, temperature, max_tokens, system_prompt)
        elif self.provider == "google":
            resp = self._chat_google(messages, model_name, temperature, max_tokens, system_prompt)
        elif self.provider == "cerebras":
            resp = self._chat_cerebras(messages, model_name, temperature, max_tokens, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        latency_ms = int((time.time() - start_time) * 1000)
        resp.metadata.update({
            "provider": self.provider,
            "model": model_name,
            "task_type": None,
            "complexity": complexity,
            "latency_ms": latency_ms,
        })
        print(f"[LLM] {resp.metadata}")
        return resp
    
    def _chat_openrouter(self, messages, model_name, temperature, max_tokens, system_prompt) -> LLMResponse:
        """Execute chat via OpenRouter."""
        if system_prompt:
            if messages[0]['role'] != 'system':
                messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            self._record_usage(usage)
            
            return LLMResponse(
                content=content,
                model=model_name,
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            raise Exception(f"OpenRouter API Error: {str(e)}")
    
    def _chat_together(self, messages, model_name, temperature, max_tokens, system_prompt) -> LLMResponse:
        """Execute chat via Together AI."""
        if system_prompt:
            if messages[0]['role'] != 'system':
                messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        self._record_usage(usage)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            usage=usage,
            finish_reason=response.choices[0].finish_reason,
        )
    
    def _chat_google(self, messages, model_name, temperature, max_tokens, system_prompt) -> LLMResponse:
        """Execute chat via Google Gemini REST API."""
        # STRICT ENFORCEMENT: Always use gemma-3-27b-it if user didn't explicitly ask for something else
        # Even if they did, per user instruction we prioritize gemma-3-27b-it for now, or at least fallback to it.
        
        # We'll just force the model name if it's one of our internal keys or aliases
        # If it's a raw google model name passed by other means, we might respect it, but 
        # given the aggressive 429s and user request, we default aggressively.
        
        if "gemma" not in model_name:
             model_name = "gemma-3-27b-it"
        elif "gemma-3-27b" in model_name and "it" not in model_name:
             model_name = "gemma-3-27b-it"

        # Ensure no double prefix
        msg_model = model_name.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{msg_model}:generateContent?key={self.google_key}"
        
        # Prepare contents
        contents = []
        system_instruction_supported = "gemini" in model_name and "flash" in model_name # Only Gemini Flash/Pro robustly support it via API field
        
        # If model doesn't support systemInstruction (like Gemma), we prepend it
        effective_system_prompt = system_prompt if system_instruction_supported else None
        prepend_system_prompt = system_prompt if not system_instruction_supported else None

        for i, msg in enumerate(messages):
            role = "user" if msg["role"] == "user" else "model"
            if msg["role"] == "system":
                continue
                
            text = msg["content"]
            
            # Prepend system prompt to the first user message if needed
            if prepend_system_prompt and role == "user":
                text = f"{prepend_system_prompt}\n\n{text}"
                prepend_system_prompt = None # Only prepend once
                
            contents.append({
                "role": role,
                "parts": [{"text": text}]
            })
            
        # Handle case where system prompt remains but no user message found (edge case)
        if prepend_system_prompt:
             contents.insert(0, {
                 "role": "user",
                 "parts": [{"text": prepend_system_prompt}]
             })
            
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        if effective_system_prompt:
             payload["systemInstruction"] = {
                 "parts": [{"text": effective_system_prompt}]
             }
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            # Candidates -> content -> parts -> text
            try:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                finish_reason = data["candidates"][0].get("finishReason", "stop")
                
                usage_meta = data.get("usageMetadata", {})
                usage = {
                    "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                    "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                    "total_tokens": usage_meta.get("totalTokenCount", 0),
                }
                self._record_usage(usage)
                
                return LLMResponse(
                    content=content,
                    model=model_name,
                    usage=usage,
                    finish_reason=finish_reason,
                )
            except (KeyError, IndexError) as e:
                # Handle cases where safety blocks content
                if "candidates" in data and not data["candidates"][0].get("content"):
                    finish_reason = data["candidates"][0].get("finishReason", "unknown")
                    raise Exception(f"Gemini blocked content. Reason: {finish_reason}")
                raise Exception(f"Failed to parse Gemini response: {data}")
                
        except Exception as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_msg += f". Details: {e.response.text}"
                except:
                    pass
            raise Exception(f"Gemini API Error: {error_msg}")

    def _run_coroutine_sync(self, coro):
        """Run an async coroutine from sync code, even if an event loop is already running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        result: Dict[str, Any] = {}
        exception_holder: Dict[str, Any] = {}

        def runner():
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result["value"] = new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                exception_holder["exception"] = e

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        
        if "exception" in exception_holder:
            raise exception_holder["exception"]
        
        return result.get("value")

    def _chat_cerebras(self, messages, model_name, temperature, max_tokens, system_prompt) -> LLMResponse:
        """Execute chat via Cerebras using async HTTPX client with OpenAI-compatible schema."""
        if system_prompt and messages and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            content, usage, finish_reason = self._run_coroutine_sync(
                self._cerebras_call(messages=messages, model=model_name)
            )
        except Exception as e:
            raise Exception(f"Cerebras API Error: {e}")

        self._record_usage(usage)
        return LLMResponse(
            content=content,
            model=model_name,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _record_usage(self, usage: Dict[str, int]) -> None:
        """Track usage totals and per-provider breakdown."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        self.total_tokens_used += total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        provider = self.provider or "unknown"
        self.model_usage_breakdown[provider] = self.model_usage_breakdown.get(provider, 0) + total_tokens

    async def _cerebras_call(self, messages: List[Dict[str, Any]], model: str) -> Tuple[str, Dict[str, int], str]:
        """Async Cerebras call using httpx; returns content, usage, finish_reason."""
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cerebras_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else "unknown"
            detail = e.response.text if e.response is not None else str(e)
            raise Exception(f"HTTP {status}: {detail}")
        except httpx.RequestError as e:
            raise Exception(f"Request failed: {e}")

        try:
            data = response.json()
        except ValueError:
            raise Exception("Invalid JSON response from Cerebras")

        try:
            choice = data.get("choices", [])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "stop")
        except Exception:
            raise Exception(f"Malformed Cerebras response: {data}")

        if not content:
            raise Exception("Cerebras response missing content")

        usage_meta = data.get("usage", {}) or {}
        usage = {
            "prompt_tokens": usage_meta.get("prompt_tokens", 0),
            "completion_tokens": usage_meta.get("completion_tokens", 0),
            "total_tokens": usage_meta.get(
                "total_tokens",
                usage_meta.get("prompt_tokens", 0) + usage_meta.get("completion_tokens", 0),
            ),
        }

        return content, usage, finish_reason
    
    # V2: Adaptive Model Routing
    def route_model(
        self,
        task_type: str,
        model_hint: str = "medium",
        budget_remaining_ms: Optional[int] = None,
    ) -> str:
        """
        V2: Select model based on task type, hint, and budget.
        
        Args:
            task_type: Type of task (sanitize, validate, synthesize, etc.)
            model_hint: Hint from task graph (small, medium, large)
            budget_remaining_ms: Optional time budget remaining
            
        Returns:
            Model name to use
        """
        # Determine tier from task type or hint
        tier = self.TASK_ROUTING.get(task_type, model_hint)
        
        # Downgrade if budget is tight
        if budget_remaining_ms is not None and budget_remaining_ms < 2000:
            if tier == "large":
                tier = "medium"
            elif tier == "medium":
                tier = "small"
        
        # Get available models for tier
        models = self.MODEL_TIERS.get(tier, self.MODEL_TIERS["medium"])
        
        # Return first available model in tier
        # Could be enhanced with load balancing, cost optimization
        return models[0] if models else self.MODELS["default"]

    def estimate_complexity(self, query: str) -> float:
        """Heuristic complexity estimate (0–1) for routing/analytics."""
        if not query:
            return 0.0

        length_score = min(len(query) / 600.0, 1.0)
        keywords = [
            "compare",
            "tradeoff",
            "architecture",
            "design",
            "benchmark",
            "multi-step",
        ]
        keyword_hits = sum(1 for k in keywords if k.lower() in query.lower())
        keyword_score = min(keyword_hits * 0.15, 0.6)

        score = min(max(length_score * 0.5 + keyword_score, 0.0), 1.0)
        return score
    
    def chat_with_routing(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "default",
        model_hint: str = "medium",
        task_id: Optional[str] = None,
        budget_ms: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        V2: Chat with automatic model routing.
        
        Args:
            messages: Chat messages
            task_type: Type of task for routing
            model_hint: Model tier hint
            task_id: Optional task ID for budget tracking
            budget_ms: Optional time budget
            temperature: LLM temperature
            max_tokens: Max tokens
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse
        """
        # Get remaining budget for task
        remaining = self._task_budgets.get(task_id, budget_ms)
        
        # Route to appropriate model
        model = self.route_model(task_type, model_hint, remaining)
        complexity = self._complexity_from_messages(messages)
        
        # Execute chat
        start_time = time.time()
        response = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Update budget tracking
        if task_id:
            current = self._task_budgets.get(task_id, budget_ms or 60000)
            self._task_budgets[task_id] = max(0, current - elapsed_ms)
            self._task_token_usage[task_id] = (
                self._task_token_usage.get(task_id, 0) + response.usage.get("total_tokens", 0)
            )

        response.metadata.update({
            "provider": self.provider,
            "model": model,
            "task_type": task_type,
            "complexity": complexity,
            "latency_ms": elapsed_ms,
        })
        print(f"[LLM] {response.metadata}")
        
        return response
    
    def set_task_budget(self, task_id: str, budget_ms: int) -> None:
        """V2: Set budget for a task."""
        self._task_budgets[task_id] = budget_ms
    
    def get_task_usage(self, task_id: str) -> Dict[str, Any]:
        """V2: Get usage stats for a task."""
        return {
            "budget_remaining_ms": self._task_budgets.get(task_id, 0),
            "tokens_used": self._task_token_usage.get(task_id, 0),
        }
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat completion response."""
        model_name = self.MODELS.get(model, model)
        
        # Common OpenAI/OpenRouter/Together streaming logic
        if self.provider in ["openrouter", "together"]:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif self.provider == "google":
            # Simple non-streaming fallback for now
            response = self._chat_google(messages, model_name, temperature, max_tokens, system_prompt)
            yield response.content

    def _complexity_from_messages(self, messages: List[Dict[str, Any]]) -> float:
        """Extract a user-visible text and estimate complexity."""
        if not messages:
            return 0.0
        # Prefer first user message; fallback to joined content
        user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
        text = user_msgs[0] if user_msgs else " ".join([m.get("content", "") for m in messages])
        return self.estimate_complexity(text)

    def simple_query(
        self,
        query: str,
        model: str = "default",
        system_prompt: Optional[str] = None,
    ) -> str:
        """Simple single-turn query."""
        messages = [{"role": "user", "content": query}]
        response = self.chat(messages, model=model, system_prompt=system_prompt)
        return response.content
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_tokens": self.total_tokens_used,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": self.total_cost,
            "model_usage_breakdown": dict(self.model_usage_breakdown),
        }


# Convenience function for quick queries
def query_llm(
    query: str,
    model: str = "default",
    system_prompt: Optional[str] = None,
) -> str:
    """Quick utility function for single LLM queries."""
    client = LLMClient()
    return client.simple_query(query, model=model, system_prompt=system_prompt)
