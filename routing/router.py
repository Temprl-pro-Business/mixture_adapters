"""
Module for semantic-based routing of queries to appropriate adapters.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.embeddings import EmbeddingsGenerator
from .route import AdapterRoute

class SemanticRouter:
    """
    Routes user queries to appropriate adapters based on semantic similarity.
    """
    
    def __init__(self, embedding_model_name: str, similarity_threshold: float = 0.68):
        """
        Initialize the semantic router.
        
        Args:
            embedding_model_name (str): Name of the model to use for embeddings
            similarity_threshold (float): Base similarity threshold (will be adjusted dynamically)
        """
        self.embeddings_generator = EmbeddingsGenerator(embedding_model_name)
        self.base_threshold = similarity_threshold
        self.route_embeddings: Dict[str, List[np.ndarray]] = {}
        self.default_adapter_name = "base"
        self.historical_scores: List[float] = []  # Track historical scores
        self.score_window = 10  # Number of scores to keep for average
        
    def add_route(self, route: AdapterRoute) -> None:
        """
        Add a new route with its training utterances.
        
        Args:
            route (AdapterRoute): Route to add with its training utterances
        """
        if route.training_utterances:
            self.route_embeddings[route.adapter_name] = (
                self.embeddings_generator.batch_generate_embeddings(route.training_utterances)
            )
            
    def add_routes(self, routes: List[AdapterRoute]) -> None:
        """
        Add multiple routes at once.
        
        Args:
            routes (List[AdapterRoute]): List of routes to add
        """
        for route in routes:
            self.add_route(route)
            
    def calculate_similarities(self, query: str) -> Dict[str, float]:
        """
        Calculate similarity scores between query and all routes.
        
        Args:
            query (str): User's input query
            
        Returns:
            Dict[str, float]: Dictionary mapping adapter names to similarity scores
        """
        query_embedding = self.embeddings_generator.generate_embedding(query)
        similarities: Dict[str, float] = {}
        
        for adapter_name, embeddings in self.route_embeddings.items():
            if not embeddings:
                continue
                
            # Calculate similarity with each example utterance
            utterance_similarities = [
                self.embeddings_generator.calculate_cosine_similarity(query_embedding, emb)
                for emb in embeddings
            ]
            
            # Use mean similarity as the score for this adapter
            similarities[adapter_name] = float(np.mean(utterance_similarities))
            
        return similarities
            
    def calculate_dynamic_threshold(self, similarities: Dict[str, float]) -> float:
        """
        Calculate a dynamic threshold based on historical and current scores.
        
        Args:
            similarities (Dict[str, float]): Current similarity scores
            
        Returns:
            float: Dynamic threshold value
        """
        if not similarities:
            return self.base_threshold
            
        # Calculate mean of current scores
        current_mean = np.mean(list(similarities.values()))
        
        # Update historical scores
        self.historical_scores.append(current_mean)
        if len(self.historical_scores) > self.score_window:
            self.historical_scores.pop(0)
            
        # Calculate dynamic threshold
        historical_mean = np.mean(self.historical_scores)
        dynamic_threshold = (historical_mean + self.base_threshold) / 2
        
        return dynamic_threshold
            
    def route_query(self, query: str) -> str:
        """
        Route a user query to the most appropriate adapter.
        
        Args:
            query (str): User's input query
            
        Returns:
            str: Name of the most appropriate adapter
        """
        adapter_name, _ = self.route_query_with_scores(query)
        return adapter_name
        
    def route_query_with_scores(self, query: str) -> Tuple[str, Dict[str, float]]:
        """
        Route a query and return both the selected adapter and similarity scores.
        
        Args:
            query (str): User's input query
            
        Returns:
            Tuple[str, Dict[str, float]]: Selected adapter name and all similarity scores
        """
        similarities = self.calculate_similarities(query)
        
        # If no routes or no similarities
        if not similarities:
            similarities[self.default_adapter_name] = 0.0
            return self.default_adapter_name, similarities
            
        # Calculate dynamic threshold
        dynamic_threshold = self.calculate_dynamic_threshold(similarities)
        
        # Get highest scoring adapter
        best_adapter, best_score = max(similarities.items(), key=lambda x: x[1])
        
        # Add base model score for logging
        similarities[self.default_adapter_name] = 0.0
        
        # Check if best score meets dynamic threshold
        if best_score < dynamic_threshold:
            return self.default_adapter_name, similarities
            
        return best_adapter, similarities 
        return selected_adapter, similarities 