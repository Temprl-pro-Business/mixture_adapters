"""
Main module for the mixture of adapters system.
"""

import asyncio
import logging
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncGenerator
from .config.settings import Settings
from .config.adapter_config import AdapterConfigLoader, AdapterConfig
from .core.model_manager import ModelManager
from .core.adapter_manager import AdapterManager
from .core.chat_generator import ChatGenerator
from .routing.router import SemanticRouter
from .utils.logger import ColoredLogger
from .api.server import APIServer
import numpy as np

class MixtureOfAdapters:
    """
    Main class that orchestrates the mixture of adapters system.
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_config_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        api_server: bool = False,
        api_host: str = "0.0.0.0",
        api_port: int = 8000
    ):
        """
        Initialize the mixture of adapters system.
        
        Args:
            config_path (Optional[Union[str, Path]]): Path to the adapter configuration file
            model_config_path (Optional[Union[str, Path]]): Path to the model configuration file
            verbose (bool): Whether to print detailed information about adapter selection and generation
            api_server (bool): Whether to start the OpenAI-compatible API server
            api_host (str): Host to bind the API server to
            api_port (int): Port for the API server to listen on
        """
        self.verbose = verbose
        self.logger = ColoredLogger(__name__, level=logging.INFO if verbose else logging.WARNING)
        
        # Load configurations
        self.config_loader = AdapterConfigLoader()
        self.adapter_config = self._load_config(config_path)
        self.model_config = self._load_model_config(model_config_path)
        
        # Initialize model manager with config
        model_settings = self.model_config["model_settings"]["base_model"]
        self.model_manager = ModelManager(
            model_name=model_settings["name"],
            load_in_8bit=model_settings.get("load_in_8bit", False),
            load_in_4bit=model_settings.get("load_in_4bit", True)
        )
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(self.model_manager.get_model())
        
        # Initialize semantic router with config
        embedding_settings = self.model_config["model_settings"]["embedding_model"]
        self.router = SemanticRouter(
            embedding_model_name=embedding_settings["name"],
            similarity_threshold=embedding_settings.get("similarity_threshold", 0.7)
        )
        
        # Load adapters and routes
        self._load_adapters()
        
        # Initialize chat generator with config
        self.chat_generator = ChatGenerator(
            model=self.adapter_manager.get_model(),
            tokenizer=self.model_manager.get_tokenizer(),
            **self.model_config["generation_settings"]
        )
        
        # Track current adapter
        self.current_adapter = None
        
        # Start API server if requested
        self.api_server = None
        if api_server:
            self.start_api_server(api_host, api_port)
            
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Start the OpenAI-compatible API server.
        
        Args:
            host (str): Host to bind to
            port (int): Port to listen on
        """
        self.logger.info(f"<LOADING>Starting API server on {host}:{port}...</LOADING>", highlight=True)
        self.api_server = APIServer(self, host=host, port=port)
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=self.api_server.start, daemon=False)
        server_thread.start()
        
        self.logger.success(f"API server running at http://{host}:{port}")
        self.logger.info("Available endpoints:", highlight=True)
        self.logger.info("  - <ADAPTER>POST /v1/chat/completions</ADAPTER>", highlight=True)
        self.logger.info("  - <ADAPTER>GET /v1/models</ADAPTER>", highlight=True)
        self.logger.info(f"\nSwagger UI available at http://{host}:{port}/docs", highlight=True)
        
    def stop_api_server(self) -> None:
        """Stop the API server if it's running."""
        if self.api_server:
            self.logger.info("<LOADING>Stopping API server...</LOADING>", highlight=True)
            # TODO: Implement graceful shutdown
            self.api_server = None
            self.logger.success("API server stopped")

    def _load_model_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Load model configuration from file.
        
        Args:
            config_path (Optional[Union[str, Path]]): Path to the configuration file
            
        Returns:
            Dict: Model configuration
        """
        if config_path is None:
            config_path = Path.cwd() / "model_config.json"
            
        if not Path(config_path).exists():
            self.logger.warning(f"No model configuration file found at {config_path}")
            self.logger.info("Using default model configuration...")
            return {
                "model_settings": {
                    "base_model": {
                        "name": Settings.BASE_MODEL_NAME,
                        "load_in_8bit": Settings.LOAD_IN_8BIT,
                        "load_in_4bit": Settings.LOAD_IN_4BIT
                    },
                    "embedding_model": {
                        "name": Settings.EMBEDDING_MODEL_NAME,
                        "similarity_threshold": Settings.SIMILARITY_THRESHOLD
                    }
                },
                "generation_settings": Settings.get_generation_settings()
            }
            
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> AdapterConfig:
        """
        Load adapter configuration from file.
        
        Args:
            config_path (Optional[Union[str, Path]]): Path to the configuration file
            
        Returns:
            AdapterConfig: Loaded configuration
        """
        if config_path is None:
            config_path = self.config_loader.get_default_config_path()
            if not Path(config_path).exists():
                self.logger.warning(f"No configuration file found at {config_path}")
                self.logger.info("Creating example configuration file...")
                self.config_loader.create_example_config(config_path)
                self.logger.info(f"Please edit {config_path} with your adapter configurations")
                raise FileNotFoundError(f"Please configure your adapters in {config_path}")
                
        return self.config_loader.load_from_file(config_path)
        
    def _load_adapters(self) -> None:
        """Load adapters from both HuggingFace Hub and local directories."""
        routes = []
        
        # Load adapters from HuggingFace Hub
        if self.adapter_config.hub_adapters:
            self.logger.info("<LOADING>Loading adapters from HuggingFace Hub...</LOADING>", highlight=True)
            hub_routes = self.adapter_manager.load_adapters_from_hub({
                adapter.name: adapter.repo_id
                for adapter in self.adapter_config.hub_adapters
            })
            routes.extend(hub_routes)
            self.logger.success(f"Loaded {len(hub_routes)} adapters from Hub:")
            for adapter in self.adapter_config.hub_adapters:
                self.logger.info(f"  - <ADAPTER>{adapter.name}</ADAPTER> (<LOADING>{adapter.repo_id}</LOADING>)", highlight=True)
            
        # Load adapters from local directories
        if self.adapter_config.local_adapters:
            self.logger.info("<LOADING>Loading local adapters...</LOADING>", highlight=True)
            for adapter in self.adapter_config.local_adapters:
                route = self.adapter_manager.load_adapter_from_directory(
                    adapter.path,
                    adapter.name
                )
                if route is not None:
                    routes.append(route)
                    self.logger.info(f"  - <ADAPTER>{adapter.name}</ADAPTER> (<LOADING>{adapter.path}</LOADING>)", highlight=True)
            self.logger.success(f"Loaded {len(self.adapter_config.local_adapters)} local adapters")
            
        # Add routes to router
        if routes:
            self.logger.info(f"<LOADING>Adding {len(routes)} routes to semantic router</LOADING>", highlight=True)
            self.router.add_routes(routes)
        else:
            self.logger.warning("<ERROR>No adapters loaded</ERROR>", highlight=True)
            
    def _log_routing_decision(self, query: str, adapter_name: str, similarities: Dict[str, float]) -> None:
        """
        Log detailed information about the routing decision.
        
        Args:
            query (str): The input query
            adapter_name (str): The selected adapter name
            similarities (Dict[str, float]): Similarity scores for each adapter
        """
        self.logger.info("\nRouting Decision:", highlight=True)
        self.logger.info(f"Query: <QUERY>{query}</QUERY>", highlight=True)
        
        # Get dynamic threshold
        dynamic_threshold = self.router.calculate_dynamic_threshold(similarities)
        self.logger.info(f"\nDynamic Threshold: <SCORE>{dynamic_threshold:.4f}</SCORE>", highlight=True)
        
        # Calculate mean similarity
        mean_similarity = np.mean([s for s in similarities.values() if s > 0])
        self.logger.info(f"Mean Similarity: <SCORE>{mean_similarity:.4f}</SCORE>", highlight=True)
        
        self.logger.info("\nSimilarity Scores:", highlight=True)
        
        # Sort similarities by score in descending order
        sorted_scores = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_scores:
            score_text = f"{score:.4f}"
            if name == adapter_name:
                self.logger.info(
                    f"  - <ADAPTER>{name}</ADAPTER>: <SCORE>{score_text}</SCORE> <SELECTED>(SELECTED)</SELECTED>",
                    highlight=True
                )
            else:
                self.logger.info(
                    f"  - <ADAPTER>{name}</ADAPTER>: <SCORE>{score_text}</SCORE>",
                    highlight=True
                )
                
        self.logger.info(f"\nSelected Adapter: <SELECTED>{adapter_name}</SELECTED>", highlight=True)
        
    async def generate_response(
        self,
        query: str,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response for the given query using the appropriate adapter.
        
        Args:
            query (str): The user's query
            messages (List[Dict[str, str]]): The conversation history
            stream (bool): Whether to stream the response
            
        Yields:
            str: Generated text chunks if streaming, or complete response
        """
        # Route the query to the appropriate adapter
        adapter_name, similarities = self.router.route_query_with_scores(query)
        if self.verbose:
            self._log_routing_decision(query, adapter_name, similarities)
        
        # If using base model, disable all adapters
        if adapter_name == "base":
            self.adapter_manager.disable_all_adapters()
        else:
            # Set the active adapter
            self.adapter_manager.set_active_adapter(adapter_name)
        
        self.current_adapter = adapter_name
        
        # Generate the response
        self.logger.info("\n<LOADING>Generating response...</LOADING>", highlight=True)
        async for chunk in self.chat_generator.generate_chat_completion(messages, stream):
            if stream:
                print(chunk, end="", flush=True)
            yield chunk
            
        print()  # Add newline after response
        self.logger.success(f"\nResponse generated using adapter: {self.current_adapter}")
        
    def get_current_adapter(self) -> Optional[str]:
        """Get the name of the currently active adapter."""
        return self.current_adapter

async def main():
    """Main function to demonstrate the mixture of adapters system."""
    try:
        # Initialize the system with configuration, verbose output, and API server
        system = MixtureOfAdapters(
            verbose=True,
            api_server=True,  # Enable API server
            api_port=8000     # Use port 8000
        )
        
        # Create an event to signal shutdown
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            print("\nShutting down server...")
            shutdown_event.set()
            system.stop_api_server()
        
        # Handle Ctrl+C
        try:
            import signal
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        except NotImplementedError:
            # Windows doesn't support SIGINT/SIGTERM handlers
            pass
        
        # Keep the program running until shutdown is signaled
        try:
            await shutdown_event.wait()
        except KeyboardInterrupt:
            signal_handler()
            
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please configure your adapters and try again.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())