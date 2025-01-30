import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .base_agent import BaseClusteringAgent

class ClusteringAgent(BaseClusteringAgent):
    """Agent responsible for performing clustering analysis."""
    
    def __init__(self):
        super().__init__("clusterer")
        self.kmeans = None
        self.min_clusters = 2
        self.max_clusters = 10
    
    def _calculate_elbow_scores(self, data: np.ndarray) -> List[float]:
        """Calculate inertia scores for different numbers of clusters."""
        scores = []
        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            scores.append(kmeans.inertia_)
        return scores
    
    def _calculate_silhouette_scores(self, data: np.ndarray) -> List[float]:
        """Calculate silhouette scores for different numbers of clusters."""
        scores = []
        for k in range(self.min_clusters, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)
        return scores
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find optimal number of clusters using both elbow and silhouette methods."""
        elbow_scores = self._calculate_elbow_scores(data)
        silhouette_scores = self._calculate_silhouette_scores(data)
        
        # Normalize scores to combine them
        norm_elbow = 1 - (elbow_scores - np.min(elbow_scores)) / (np.max(elbow_scores) - np.min(elbow_scores))
        norm_silhouette = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))
        
        # Combine scores (higher is better)
        combined_scores = 0.5 * norm_elbow + 0.5 * norm_silhouette
        optimal_k = self.min_clusters + np.argmax(combined_scores)
        
        return optimal_k
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clustering analysis on the engineered data.
        
        Args:
            data: Dictionary containing engineered data and feature information
            
        Returns:
            Dictionary containing:
                - cluster_labels: Array of cluster assignments
                - cluster_centers: Array of cluster centers
                - optimal_n_clusters: Optimal number of clusters
                - evaluation_metrics: Dictionary of clustering evaluation metrics
        """
        df = data['engineered_data']
        feature_names = data['feature_names']
        
        # Find optimal number of clusters
        optimal_k = self._find_optimal_clusters(df.values)
        
        # Fit final clustering model
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = self.kmeans.fit_predict(df.values)
        
        # Calculate cluster centers in original feature space
        centers = pd.DataFrame(
            self.kmeans.cluster_centers_,
            columns=feature_names
        )
        
        # Calculate silhouette score for final clustering
        silhouette_avg = silhouette_score(df.values, labels)
        
        return {
            'cluster_labels': labels.tolist(),
            'cluster_centers': centers.to_dict(orient='records'),
            'optimal_n_clusters': optimal_k,
            'evaluation_metrics': {
                'silhouette_score': silhouette_avg,
                'inertia': self.kmeans.inertia_
            },
            'feature_names': feature_names,
            'data': df
        }
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine the next agent based on clustering results."""
        return "cluster_namer" 