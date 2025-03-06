"""
Module for handling text embeddings generation and similarity calculations.
"""

import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List

class EmbeddingsGenerator:
    """
    Handles the generation of text embeddings and similarity calculations.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the embeddings generator with a specific model.
        
        Args:
            model_name (str): Name of the HuggingFace model to use for embeddings
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for input text.
        
        Args:
            text (str): Input text to generate embedding for
            
        Returns:
            np.ndarray: Embedding vector
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
        
    @staticmethod
    def calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1 (np.ndarray): First vector
            vector2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score
        """
        return float(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        
    def batch_generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        return [self.generate_embedding(text) for text in texts] 