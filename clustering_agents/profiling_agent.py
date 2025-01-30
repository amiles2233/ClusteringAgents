import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_agent import BaseClusteringAgent

class DataProfilingAgent(BaseClusteringAgent):
    """Agent responsible for analyzing and profiling input data."""
    
    def __init__(self):
        super().__init__("data_profiler")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Profile the input data and identify variable types and characteristics.
        
        Args:
            data: Dictionary containing the input DataFrame under 'data' key
            
        Returns:
            Dictionary containing:
                - numerical_columns: List of numerical column names
                - categorical_columns: List of categorical column names
                - missing_values: Dictionary of missing value counts per column
                - statistics: Basic statistics for numerical columns
                - cardinality: Dictionary of unique value counts for categorical columns
        """
        df = data['data']
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            
        # Identify column types
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analyze missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Calculate basic statistics for numerical columns
        statistics = {}
        if numerical_columns:
            statistics = df[numerical_columns].describe().to_dict()
            
        # Calculate cardinality for categorical columns
        cardinality = {}
        for col in categorical_columns:
            cardinality[col] = df[col].nunique()
            
        return {
            'numerical_columns': numerical_columns,
            'categorical_columns': categorical_columns,
            'missing_values': missing_values,
            'statistics': statistics,
            'cardinality': cardinality,
            'original_data': df
        }
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine the next agent based on profiling results."""
        return "preprocessor" 