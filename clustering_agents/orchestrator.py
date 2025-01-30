import traceback
import pandas as pd
from typing import Dict, Any
from langgraph.graph import Graph

class ClusteringOrchestrator:
    def __init__(self):
        """Initialize the orchestrator with all required agents."""
        from clustering_agents.profiling_agent import DataProfilingAgent
        from clustering_agents.preprocessing_agent import PreprocessingAgent
        from clustering_agents.feature_engineering_agent import FeatureEngineeringAgent
        from clustering_agents.clustering_agent import ClusteringAgent
        from clustering_agents.cluster_naming_agent import ClusterNamingAgent
        
        self.agents = {
            "data_profiler": DataProfilingAgent(),
            "preprocessor": PreprocessingAgent(),
            "feature_engineer": FeatureEngineeringAgent(),
            "clusterer": ClusteringAgent(),
            "cluster_namer": ClusterNamingAgent()
        }
    
    async def process_agent(
        self,
        state: Dict[str, Any],
        agent_name: str,
        agents: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process the current agent and update the state."""
        print(f"Processing agent: {agent_name}")
        print(f"Input state keys: {state['data'].keys() if isinstance(state.get('data'), dict) else 'data is not a dict'}")
        
        current_agent = agents[agent_name]
        try:
            result = await current_agent.process(state["data"])
            print(f"Agent {agent_name} result keys: {result.keys() if isinstance(result, dict) else 'result is not a dict'}")
            
            # Special handling for the final result from cluster_namer
            if agent_name == "cluster_namer":
                final_result = {
                    'cluster_names': result['cluster_names'],
                    'cluster_descriptions': result['cluster_descriptions'],
                    'cluster_labels': result['cluster_labels'],
                    'evaluation_metrics': result['evaluation_metrics']
                }
                return {
                    "data": result,
                    "current_agent": "END",
                    "final_result": final_result
                }
            
            return {
                "data": result,
                "current_agent": agent_name,
                "final_result": None
            }
            
        except Exception as e:
            print(f"Error in {agent_name}: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _create_workflow(self) -> Graph:
        """Create the workflow graph using Langgraph."""
        workflow = Graph()
        
        # Add nodes for each agent
        for agent_name in self.agents:
            def create_node_function(agent=agent_name):
                async def node_function(state):
                    result = await self.process_agent(state, agent, self.agents)
                    return result
                return node_function
            workflow.add_node(agent_name, create_node_function(agent_name))
        
        # Add END node that preserves the final state
        def end_node(state):
            print("END node received state:", state)
            # Ensure we have the expected structure from cluster_namer
            if isinstance(state, dict) and 'data' in state:
                if isinstance(state['data'], dict) and 'cluster_names' in state['data']:
                    final_result = {
                        'cluster_names': state['data']['cluster_names'],
                        'cluster_descriptions': state['data']['cluster_descriptions'],
                        'cluster_labels': state['data'].get('cluster_labels', []),
                        'evaluation_metrics': state['data'].get('evaluation_metrics', {})
                    }
                    return {
                        "data": state['data'],
                        "current_agent": "END",
                        "final_result": final_result
                    }
            # Log if the expected structure is not found
            print("END node did not find expected structure in state.")
            return {
                "data": state.get('data', {}),
                "current_agent": "END",
                "final_result": None
            }
        
        workflow.add_node("END", end_node)
        
        # Define the edges
        edges = [
            ("data_profiler", "preprocessor"),
            ("preprocessor", "feature_engineer"),
            ("feature_engineer", "clusterer"),
            ("clusterer", "cluster_namer"),
            ("cluster_namer", "END")
        ]
        
        # Add edges
        for source, target in edges:
            workflow.add_edge(source, target)
        
        # Set the entry point
        workflow.set_entry_point("data_profiler")
        
        return workflow.compile()
    
    async def run_clustering_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the clustering analysis workflow."""
        print("Starting clustering analysis...")
        
        # Initialize the state
        initial_state = {
            "data": {"data": data},
            "current_agent": "data_profiler",
            "final_result": None
        }
        print("Initial state:", initial_state)
        
        # Create and run the workflow
        workflow = self._create_workflow()
        print("Workflow created, running batch...")
        
        try:
            results = await workflow.abatch([initial_state])
            print("Raw workflow results:", results)
            
            if not results or len(results) == 0:
                raise RuntimeError("No results returned from workflow")
            
            final_state = results[0]
            if not isinstance(final_state, dict) or 'final_result' not in final_state:
                print("Final state does not contain 'final_result':", final_state)
                raise RuntimeError("Invalid final state structure")
            
            return final_state['final_result']
            
        except Exception as e:
            print(f"Error during workflow execution: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise 