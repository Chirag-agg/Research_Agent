"""
Base Agent class for Deep Research Agent

Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class AgentResult:
    """Standard result format for all agents"""
    success: bool
    content: Any
    agent_name: str
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the DRA system.
    
    All agents must implement:
    - execute(): Main execution logic
    - get_system_prompt(): Agent-specific system prompt
    """
    
    def __init__(self, llm_client, context_manager=None):
        """
        Initialize the agent.
        
        Args:
            llm_client: LLMClient instance for inference
            context_manager: Optional ContextManager for session state
        """
        self.llm = llm_client
        self.context = context_manager
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main task.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            AgentResult with execution results
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    def log(self, message: str) -> None:
        """Log a message with agent name prefix."""
        print(f"[{self.name}] {message}")
