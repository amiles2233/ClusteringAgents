import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from .base_agent import BaseClusteringAgent

class FeatureEngineeringAgent(BaseClusteringAgent):
    """Agent responsible for feature engineering and transformation."""
    
    def __init__(self):
        super().__init__("feature_engineer")
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.pca = None
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features by encoding categorical variables and reducing dimensionality.
        
        Args:
            data: Dictionary containing preprocessed data and column information
            
        Returns:
            Dictionary containing:
                - engineered_data: DataFrame with engineered features
                - feature_names: List of final feature names
                - transformation_info: Information about the transformations applied
        """
        df = data['processed_data']
        numerical_columns = data['numerical_columns']
        categorical_columns = data['categorical_columns']
        
        # One-hot encode categorical variables
        if categorical_columns:
            categorical_encoded = self.onehot_encoder.fit_transform(df[categorical_columns])
            categorical_feature_names = self.onehot_encoder.get_feature_names_out(categorical_columns)
            
            # Create new dataframe with encoded features
            encoded_df = pd.DataFrame(
                categorical_encoded,
                columns=categorical_feature_names,
                index=df.index
            )
            
            # Combine with numerical features
            if numerical_columns:
                final_df = pd.concat([df[numerical_columns], encoded_df], axis=1)
            else:
                final_df = encoded_df
        else:
            final_df = df[numerical_columns].copy()
        
        # Apply PCA if the number of features is high
        n_components = None
        explained_variance_ratio = None
        if final_df.shape[1] > 10:
            n_components = min(final_df.shape[1], 10)
            self.pca = PCA(n_components=n_components)
            transformed_data = self.pca.fit_transform(final_df)
            explained_variance_ratio = self.pca.explained_variance_ratio_.tolist()
            
            # Create new dataframe with PCA components
            final_df = pd.DataFrame(
                transformed_data,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df.index
            )
        
        return {
            'engineered_data': final_df,
            'feature_names': final_df.columns.tolist(),
            'transformation_info': {
                'n_categorical_features': len(categorical_columns) if categorical_columns else 0,
                'n_numerical_features': len(numerical_columns) if numerical_columns else 0,
                'pca_components': n_components,
                'explained_variance_ratio': explained_variance_ratio
            }
        }
    
    def get_next_agent(self, result: Dict[str, Any]) -> str:
        """Determine the next agent based on feature engineering results."""
        return "clusterer" 