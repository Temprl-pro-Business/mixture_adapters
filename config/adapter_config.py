"""
Module for loading and validating user adapter configurations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from jsonschema import validate, ValidationError

@dataclass
class HubAdapter:
    """Configuration for a HuggingFace Hub adapter."""
    name: str
    repo_id: str

@dataclass
class LocalAdapter:
    """Configuration for a local adapter."""
    name: str
    path: str

@dataclass
class AdapterConfig:
    """Complete adapter configuration."""
    hub_adapters: List[HubAdapter]
    local_adapters: List[LocalAdapter]

class AdapterConfigLoader:
    """
    Handles loading and validation of user adapter configurations.
    """
    
    def __init__(self):
        """Initialize the config loader."""
        self.schema = self._load_schema()
        
    @staticmethod
    def _load_schema() -> Dict:
        """Load the JSON schema for validation."""
        schema_path = Path(__file__).parent / "adapter_config_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    def _validate_config(self, config: Dict) -> None:
        """
        Validate the configuration against the schema.
        
        Args:
            config (Dict): Configuration to validate
            
        Raises:
            ValidationError: If the configuration is invalid
        """
        try:
            validate(instance=config, schema=self.schema)
        except ValidationError as e:
            raise ValidationError(f"Invalid adapter configuration: {str(e)}")
            
    def _validate_local_paths(self, config: Dict) -> None:
        """
        Validate that local adapter paths exist.
        
        Args:
            config (Dict): Configuration to validate
            
        Raises:
            FileNotFoundError: If a local adapter path doesn't exist
        """
        for adapter in config.get("adapters", {}).get("local_adapters", []):
            path = Path(adapter["path"])
            if not path.exists():
                raise FileNotFoundError(f"Local adapter path not found: {path}")
                
    def _normalize_paths(self, config: Dict) -> Dict:
        """
        Convert relative paths to absolute paths.
        
        Args:
            config (Dict): Configuration with paths to normalize
            
        Returns:
            Dict: Configuration with normalized paths
        """
        for adapter in config.get("adapters", {}).get("local_adapters", []):
            path = Path(adapter["path"])
            if not path.is_absolute():
                adapter["path"] = str(Path.cwd() / path)
        return config
    
    def load_from_file(self, config_path: Union[str, Path]) -> AdapterConfig:
        """
        Load and validate adapter configuration from a file.
        
        Args:
            config_path (Union[str, Path]): Path to the configuration file
            
        Returns:
            AdapterConfig: Validated adapter configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValidationError: If the configuration is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load the configuration
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        # Validate against schema
        self._validate_config(config)
        
        # Normalize paths
        config = self._normalize_paths(config)
        
        # Validate local paths
        self._validate_local_paths(config)
        
        # Convert to dataclass
        adapters = config["adapters"]
        return AdapterConfig(
            hub_adapters=[
                HubAdapter(**adapter)
                for adapter in adapters.get("hub_adapters", [])
            ],
            local_adapters=[
                LocalAdapter(**adapter)
                for adapter in adapters.get("local_adapters", [])
            ]
        )
        
    @staticmethod
    def get_default_config_path() -> Path:
        """Get the default path for the adapter configuration file."""
        return Path.cwd() / "adapter_config.json"
        
    def create_example_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create an example configuration file.
        
        Args:
            output_path (Optional[Union[str, Path]]): Path to write the example config
        """
        example_config = {
            "adapters": {
                "hub_adapters": [
                    {
                        "name": "go_adapter",
                        "repo_id": "your-username/go-programming-adapter"
                    },
                    {
                        "name": "python_adapter",
                        "repo_id": "your-username/python-programming-adapter"
                    }
                ],
                "local_adapters": [
                    {
                        "name": "custom_adapter",
                        "path": "adapters/custom_adapter"
                    }
                ]
            }
        }
        
        output_path = output_path or self.get_default_config_path()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(example_config, f, indent=4) 