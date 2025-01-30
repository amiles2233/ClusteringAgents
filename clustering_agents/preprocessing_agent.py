import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from .base_agent import BaseClusteringAgent

class PreprocessingAgent(BaseClusteringAgent):
    """Agent responsible for data preprocessing and cleaning."""
    
    def __init__(self):
        super().__init__("preprocessor")
        self.numerical_scaler = StandardScaler()
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the data by handling missing values and scaling numerical features.
        
        Args:
            data: Dictionary containing profiling results and original data
            
        Returns:
            Dictionary containing:
                - processed_data: Cleaned and scaled DataFrame
                - scaling_info: Information about the scaling transformations
                - imputation_info: Information about the imputation steps
        """
        df = data['original_data']
        numerical_columns = data['numerical_columns']
        categorical_columns = data['categorical_columns']
        
        # Handle missing values
        if numerical_columns:
            df[numerical_columns] = self.numerical_imputer.fit_transform(df[numerical_columns])
        
        if categorical_columns:
            df[categorical_columns] = self.categorical_imputer.fit_transform(df[categorical_columns])
        
        # Scale numerical features
        if numerical_columns:
            df[numerical_columns] = self.numerical_scaler.fit_transform(df[numerical_columns])
        
        return {
            'processed_data': df,
            'numerical_columns': numerical_columns,
            'categorical_columns': categorical_columns,
            'scaling_params': {
                'mean': self.numerical_scaler.mean_.tolist() if numerical_columns else None,
                'scale': self.numerical_scaler.scale_.tolist() if numerical_columns else None
            },
            'imputation_info': {
                'numerical_strategy': 'mean',
                'categorical_strategy': 'most_frequent'
            }
        }
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine the next agent based on preprocessing results."""
        return "feature_engineer" 