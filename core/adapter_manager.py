"""
Module for managing PEFT adapters and their configurations.
"""

import json
from typing import Dict, List, Optional, Union
from pathlib import Path
from peft import PeftModel
from huggingface_hub import hf_hub_download
from transformers import PreTrainedModel
from ..routing.route import AdapterRoute
from .adapter_loader import AdapterLoader

class AdapterManager:
    """
    Manages the loading and switching of PEFT adapters for the base model.
    """
    
    def __init__(self, base_model: PreTrainedModel):
        """
        Initialize the adapter manager.
        
        Args:
            base_model (PreTrainedModel): The base model to load adapters for
        """
        self.base_model = base_model
        self.peft_model: Optional[PeftModel] = None
        self.loaded_adapters: Dict[str, str] = {}  # adapter_name -> adapter_path
        
    def load_adapter_from_hub(self, adapter_name: str, adapter_path: str) -> Optional[AdapterRoute]:
        """
        Load a single adapter from HuggingFace Hub.
        
        Args:
            adapter_name (str): Name to identify the adapter
            adapter_path (str): HuggingFace Hub path to the adapter
            
        Returns:
            Optional[AdapterRoute]: Route configuration if semantic routing is enabled
        """
        print(f"Loading adapter from Hub: {adapter_name} from {adapter_path}")
        
        # Initialize PEFT model if not already done
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name
            )
        else:
            self.peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
            
        self.loaded_adapters[adapter_name] = adapter_path
        print(f"Adapter {adapter_name} loaded successfully from Hub")
        
        # Try to load semantic routing configuration
        try:
            config_path = hf_hub_download(repo_id=adapter_path, filename="adapter_config.json")
            with open(config_path, "r") as f:
                adapter_config = json.load(f)
                
            if "semantic_routing" in adapter_config:
                utterances = adapter_config["semantic_routing"]["questions"]
                return AdapterRoute(adapter_name=adapter_name, training_utterances=utterances)
                
        except Exception as e:
            print(f"No semantic routing config found for {adapter_name}: {str(e)}")
            
        return None
        
    def load_adapter_from_directory(self, adapter_dir: Union[str, Path], adapter_name: Optional[str] = None) -> Optional[AdapterRoute]:
        """
        Load a single adapter from a local directory.
        
        Args:
            adapter_dir (Union[str, Path]): Path to the adapter directory
            adapter_name (Optional[str]): Name to identify the adapter. If None, uses directory name
            
        Returns:
            Optional[AdapterRoute]: Route configuration if semantic routing is enabled
        """
        adapter_info = AdapterLoader.load_from_directory(adapter_dir, adapter_name)
        adapter_name = adapter_info["name"]
        adapter_path = adapter_info["path"]
        
        print(f"Loading adapter from directory: {adapter_name} from {adapter_path}")
        
        # Initialize PEFT model if not already done
        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name
            )
        else:
            self.peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
            
        self.loaded_adapters[adapter_name] = adapter_path
        print(f"Adapter {adapter_name} loaded successfully from directory")
        
        return adapter_info.get("route")
        
    def load_adapters_from_directory(self, base_dir: Union[str, Path]) -> List[AdapterRoute]:
        """
        Load multiple adapters from a directory.
        
        Args:
            base_dir (Union[str, Path]): Base directory containing adapter directories
            
        Returns:
            List[AdapterRoute]: List of route configurations for adapters with semantic routing
        """
        routes = []
        adapter_infos = AdapterLoader.load_from_directories(base_dir)
        
        for adapter_info in adapter_infos:
            route = self.load_adapter_from_directory(
                adapter_info["path"],
                adapter_info["name"]
            )
            if route is not None:
                routes.append(route)
                
        return routes
        
    def load_adapters_from_hub(self, adapter_configs: Dict[str, str]) -> List[AdapterRoute]:
        """
        Load multiple adapters from HuggingFace Hub.
        
        Args:
            adapter_configs (Dict[str, str]): Dictionary mapping adapter names to their Hub paths
            
        Returns:
            List[AdapterRoute]: List of route configurations for adapters with semantic routing
        """
        routes = []
        for adapter_name, adapter_path in adapter_configs.items():
            route = self.load_adapter_from_hub(adapter_name, adapter_path)
            if route is not None:
                routes.append(route)
        return routes
        
    def set_active_adapter(self, adapter_name: str) -> None:
        """
        Set the active adapter for generation.
        
        Args:
            adapter_name (str): Name of the adapter to activate
        """
        if adapter_name != "base":
            if adapter_name not in self.loaded_adapters:
                raise ValueError(f"Adapter {adapter_name} not loaded")
            self.peft_model.set_adapter(adapter_name)
            
    def get_model(self) -> PreTrainedModel:
        """Get the current model (either base or PEFT model)."""
        return self.peft_model if self.peft_model is not None else self.base_model 