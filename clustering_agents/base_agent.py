from typing import Dict, Any
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish
from langchain_core.messages import BaseMessage

class BaseClusteringAgent:
    """Base class for all clustering agents."""
    
    def __init__(self, name: str):
        self.name = name
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return the results.
        
        Args:
            data: Dictionary containing the input data and any additional parameters
            
        Returns:
            Dictionary containing the processed results
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine the next agent in the pipeline based on the current results.
        
        Args:
            result: Dictionary containing the current processing results
            
        Returns:
            Name of the next agent to be called
        """
        raise NotImplementedError("Subclasses must implement get_next_agent method") 