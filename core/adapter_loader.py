"""
Module for loading PEFT adapters from user-provided files or directories.
"""

import os
import json
from typing import Dict, List, Optional, Union
from pathlib import Path
from ..routing.route import AdapterRoute

class AdapterLoader:
    """
    Handles loading of PEFT adapters from user-provided files or directories.
    """
    
    @staticmethod
    def validate_adapter_directory(adapter_dir: Union[str, Path]) -> bool:
        """
        Validate that a directory contains required adapter files.
        
        Args:
            adapter_dir (Union[str, Path]): Path to the adapter directory
            
        Returns:
            bool: True if directory contains required files
        """
        adapter_dir = Path(adapter_dir)
        required_files = [
            "adapter_config.json",
            "adapter_model.bin",
            "config.json"
        ]
        
        return all(adapter_dir.joinpath(file).exists() for file in required_files)
    
    @staticmethod
    def load_adapter_config(adapter_dir: Union[str, Path]) -> Dict:
        """
        Load adapter configuration from a directory.
        
        Args:
            adapter_dir (Union[str, Path]): Path to the adapter directory
            
        Returns:
            Dict: Adapter configuration
        """
        adapter_dir = Path(adapter_dir)
        config_path = adapter_dir / "adapter_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Adapter config not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    @staticmethod
    def extract_routing_config(adapter_config: Dict, adapter_name: str) -> Optional[AdapterRoute]:
        """
        Extract semantic routing configuration from adapter config.
        
        Args:
            adapter_config (Dict): Adapter configuration dictionary
            adapter_name (str): Name of the adapter
            
        Returns:
            Optional[AdapterRoute]: Route configuration if semantic routing is enabled
        """
        if "semantic_routing" in adapter_config:
            utterances = adapter_config["semantic_routing"]["questions"]
            return AdapterRoute(adapter_name=adapter_name, training_utterances=utterances)
        return None
    
    @classmethod
    def load_from_directory(cls, 
                          adapter_dir: Union[str, Path], 
                          adapter_name: Optional[str] = None) -> Dict:
        """
        Load adapter from a local directory.
        
        Args:
            adapter_dir (Union[str, Path]): Path to the adapter directory
            adapter_name (Optional[str]): Name to give the adapter. If None, uses directory name
            
        Returns:
            Dict: Dictionary containing adapter information
        """
        adapter_dir = Path(adapter_dir)
        if not cls.validate_adapter_directory(adapter_dir):
            raise ValueError(f"Invalid adapter directory: {adapter_dir}")
            
        if adapter_name is None:
            adapter_name = adapter_dir.name
            
        adapter_config = cls.load_adapter_config(adapter_dir)
        route_config = cls.extract_routing_config(adapter_config, adapter_name)
        
        return {
            "name": adapter_name,
            "path": str(adapter_dir),
            "config": adapter_config,
            "route": route_config
        }
    
    @classmethod
    def load_from_directories(cls, base_dir: Union[str, Path]) -> List[Dict]:
        """
        Load all adapters from subdirectories of a base directory.
        
        Args:
            base_dir (Union[str, Path]): Base directory containing adapter directories
            
        Returns:
            List[Dict]: List of loaded adapter configurations
        """
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise NotADirectoryError(f"Directory not found: {base_dir}")
            
        adapters = []
        for dir_path in base_dir.iterdir():
            if dir_path.is_dir() and cls.validate_adapter_directory(dir_path):
                try:
                    adapter_info = cls.load_from_directory(dir_path)
                    adapters.append(adapter_info)
                except Exception as e:
                    print(f"Error loading adapter from {dir_path}: {str(e)}")
                    
        return adapters 