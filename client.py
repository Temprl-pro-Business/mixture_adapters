"""
Module for the MixtureOfAdapters client.

This module provides a client interface for interacting with a MixtureOfAdapters server,
which enables dynamic routing and mixing of multiple language model adapters.

Key Features:
- Semantic routing between multiple adapters
- Streaming and non-streaming response generation
- Support for chat-style message formats
"""

import json
import logging
import requests
import sseclient
from typing import List, Dict, Optional, Generator, Union

# Set up logging
logger = logging.getLogger(__name__)

class MixtureClient:
    """
    Client for interacting with the MixtureOfAdapters server.
    
    Provides methods to:
    - List available models and adapters
    - Generate responses using semantic routing
    - Stream responses chunk by chunk
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url (str): Base URL of the MixtureOfAdapters server. 
                          Defaults to localhost:8000.
        """
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized MixtureClient with base URL: {self.base_url}")
        
    def list_models(self) -> Dict:
        """
        List available models and adapters.
        
        Returns:
            Dict: Dictionary containing available models and their configurations
        
        Raises:
            requests.exceptions.RequestException: If the server request fails
        """
        logger.debug("Fetching available models")
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        models = response.json()
        logger.info(f"Found {len(models)} available models")
        return models
        
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = True,
        **kwargs
    ) -> Union[Dict, Generator[str, None, None]]:
        """
        Generate a response from the model.
        
        Args:
            messages (List[Dict[str, str]]): List of messages in the conversation
            model (Optional[str]): Model/adapter to use (will use semantic routing if not specified)
            stream (bool): Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Union[Dict, Generator[str, None, None]]: Response text or stream of chunks
            
        Raises:
            requests.exceptions.RequestException: If the server request fails
        """
        url = f"{self.base_url}/generate"
        
        data = {
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if model:
            data["model"] = model
            logger.info(f"Using specified model: {model}")
        else:
            logger.info("Using semantic routing to select model")
            
        if stream:
            logger.debug("Starting streaming response")
            return self._stream_response(url, data)
        else:
            logger.debug("Generating complete response")
            return self._generate_complete(url, data)
            
    def _stream_response(self, url: str, data: Dict) -> Generator[str, None, None]:
        """
        Stream response chunks.
        
        Args:
            url (str): API endpoint URL
            data (Dict): Request payload
            
        Yields:
            str: Response content chunks
            
        Raises:
            requests.exceptions.RequestException: If the server request fails
        """
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data and event.data.strip():
                event.data = event.data.replace("data: ", "")
                try:
                    chunk = json.loads(event.data)
                    if chunk.get("content"):
                        yield chunk["content"]
                except json.JSONDecodeError:
                    logger.warning("Received malformed JSON event, skipping")
                    continue
                    
    def _generate_complete(self, url: str, data: Dict) -> Dict:
        """
        Get complete response.
        
        Args:
            url (str): API endpoint URL
            data (Dict): Request payload
            
        Returns:
            Dict: Complete response from the model
            
        Raises:
            requests.exceptions.RequestException: If the server request fails
        """
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()