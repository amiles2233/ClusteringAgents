import asyncio
import pandas as pd
import numpy as np
import traceback
from dotenv import load_dotenv
from clustering_agents import ClusteringOrchestrator

async def main():
    # Load environment variables (for OpenAI API key)
    load_dotenv()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with clear clusters
    data = {
        'income': np.concatenate([
            np.random.normal(30000, 5000, n_samples // 3),
            np.random.normal(60000, 8000, n_samples // 3),
            np.random.normal(100000, 15000, n_samples // 3)
        ]),
        'age': np.concatenate([
            np.random.normal(25, 3, n_samples // 3),
            np.random.normal(40, 5, n_samples // 3),
            np.random.normal(55, 7, n_samples // 3)
        ]),
        'spending_score': np.concatenate([
            np.random.normal(30, 10, n_samples // 3),
            np.random.normal(50, 15, n_samples // 3),
            np.random.normal(70, 20, n_samples // 3)
        ]),
        'education_years': np.concatenate([
            np.random.normal(12, 1, n_samples // 3),
            np.random.normal(16, 2, n_samples // 3),
            np.random.normal(18, 2, n_samples // 3)
        ])
    }
    
    # Add some categorical variables
    data['occupation'] = np.random.choice(
        ['student', 'professional', 'retired'],
        size=n_samples-1,
        p=[0.3, 0.5, 0.2]
    )
    data['location'] = np.random.choice(
        ['urban', 'suburban', 'rural'],
        size=n_samples-1,
        p=[0.6, 0.3, 0.1]
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize the orchestrator
    orchestrator = ClusteringOrchestrator()
    
    # Run the clustering analysis
    print("Starting clustering analysis...")
    result = await orchestrator.run_clustering_analysis(df)
    
    # Print results
    print("\nClustering Analysis Results:")
    print("-" * 50)
    print(f"Number of clusters: {len(result['cluster_names'])}")
    print("\nCluster Names and Descriptions:")
    for i, (name, desc) in enumerate(zip(result['cluster_names'], result['cluster_descriptions'])):
        print(f"\nCluster {i + 1}: {name}")
        print("Characteristics:")
        print(desc)
    
    print("\nEvaluation Metrics:")
    print(f"Silhouette Score: {result['evaluation_metrics']['silhouette_score']:.3f}")
    print(f"Inertia: {result['evaluation_metrics']['inertia']:.3f}")
    
    # Create a DataFrame with cluster assignments
    df['Cluster'] = result['cluster_labels']
    df['Cluster_Name'] = [result['cluster_names'][label] for label in result['cluster_labels']]
    
    # Print cluster sizes
    print("\nCluster Sizes:")
    print(df['Cluster_Name'].value_counts())

if __name__ == "__main__":
    asyncio.run(main()) 