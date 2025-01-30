import pandas as pd
import numpy as np
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .base_agent import BaseClusteringAgent

class ClusterNamingAgent(BaseClusteringAgent):
    """Agent responsible for automatically naming clusters based on their characteristics."""
    
    def __init__(self):
        super().__init__("cluster_namer")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        self.naming_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing data clusters and providing meaningful names based on their characteristics.
            Given the cluster centers and key statistics, provide a concise but descriptive name for each cluster.
            The name should reflect the most distinctive features of the cluster."""),
            ("user", """Cluster characteristics:
            {cluster_description}
            
            Please provide a concise name (max 5 words) that captures the key characteristics of this cluster.""")
        ])
    
    def _get_cluster_description(self, cluster_center: Dict[str, float], feature_names: List[str]) -> str:
        """Generate a description of the cluster based on its center and features."""
        # Sort features by absolute value to identify most important characteristics
        sorted_features = sorted(
            [(name, abs(value)) for name, value in cluster_center.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top 5 most distinctive features
        top_features = sorted_features[:5]
        
        description = "This cluster is characterized by:\n"
        for feature, value in top_features:
            description += f"- {feature}: {cluster_center[feature]:.2f}\n"
        
        return description
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Name clusters based on their characteristics.
        
        Args:
            data: Dictionary containing clustering results and feature information
            
        Returns:
            Dictionary containing:
                - cluster_names: List of generated cluster names
                - cluster_descriptions: Detailed descriptions of each cluster
                - cluster_centers: Original cluster centers
                - cluster_labels: Original cluster labels
                - evaluation_metrics: Original evaluation metrics
        """
        print("ClusterNamingAgent processing data:", data.keys())  # Debug print
        
        cluster_centers = data.get('cluster_centers', [])
        feature_names = data.get('feature_names', [])
        cluster_labels = data.get('cluster_labels', [])
        evaluation_metrics = data.get('evaluation_metrics', {})
        
        if not cluster_centers:
            raise ValueError("No cluster centers provided in input data")
        
        cluster_names = []
        cluster_descriptions = []
        
        for i, center in enumerate(cluster_centers):
            # Generate cluster description
            description = self._get_cluster_description(center, feature_names)
            
            # Get cluster name from LLM
            chain = self.naming_prompt | self.llm
            response = await chain.ainvoke({"cluster_description": description})
            name = response.content.strip()
            
            print(f"Generated name for cluster {i}: {name}")  # Debug print
            
            cluster_names.append(name)
            cluster_descriptions.append(description)
        
        result = {
            'cluster_names': cluster_names,
            'cluster_descriptions': cluster_descriptions,
            'cluster_centers': cluster_centers,
            'cluster_labels': cluster_labels,
            'evaluation_metrics': evaluation_metrics
        }
        
        print("ClusterNamingAgent returning result:", result.keys())  # Debug print
        return result
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """This is the final agent in the pipeline."""
        return None 