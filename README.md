# Mixture of Adapters

A system for managing multiple PEFT (Parameter-Efficient Fine-Tuning) adapters with semantic routing.

## Features

- Load multiple PEFT adapters from HuggingFace Hub or local directories
- Semantic routing of queries to appropriate adapters
- Streaming chat completions
- Support for both local and remote adapters
- Automatic adapter switching based on query content
- Simple JSON configuration for adapter management

## Installation

1. Clone the repository
2. Install the requirements:```bash
pip install -r requirements.txt```

## Configuration

The system uses a JSON configuration file to manage adapters. By default, it looks for `adapter_config.json` in the current directory. If not found, it will create an example configuration file.

### Configuration File Structure

```json
{
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
```

### Hub Adapters
Configure adapters from HuggingFace Hub:
- `name`: A unique identifier for the adapter
- `repo_id`: The HuggingFace Hub repository ID

### Local Adapters
Configure adapters from your local filesystem:
- `name`: A unique identifier for the adapter
- `path`: Path to the adapter directory (absolute or relative to current directory)

## Local Adapter Directory Structure

Each local adapter directory must contain:

- `adapter_config.json`: Configuration file for the adapter
- `adapter_model.bin`: The trained adapter weights
- `config.json`: Model configuration file

The `adapter_config.json` should include semantic routing configuration:
```json
{
    "semantic_routing": {
        "questions": [
            "Is this query asking to fix '的地得' usage in Chinese?",
            "Does this involve correcting common Chinese grammar patterns like '把' or '被'?",
            "Is this about fixing punctuation marks in Chinese text?",
            "Does this involve correcting simplified/traditional character usage or typos?"
        ]
    },
    "base_model_name_or_path": "base_model_name",
    "bias": "none",
    "inference_mode": true,
    "init_lora_weights": true,
    "layers_pattern": null,
    "layers_to_transform": null,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "r": 8,
    "target_modules": ["q_proj", "v_proj"]
}
```

## Usage

### Basic Usage

```python
from mixture_adapters import MixtureOfAdapters

async def main():
    # Initialize with default config path (./adapter_config.json)
    system = MixtureOfAdapters()
    
    # Or specify a custom config path
    system = MixtureOfAdapters("path/to/config.json")
    
    # Generate a response
    response = await system.generate_response(
        "How do I use goroutines in Go?",
        [{"role": "user", "content": "How do I use goroutines in Go?"}]
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### First-Time Setup

1. Run the system once to generate an example configuration:
```python
system = MixtureOfAdapters()  # Will create example adapter_config.json
```

2. Edit the generated `adapter_config.json` with your adapter configurations

3. Run your application again with the configured adapters

### Configuration Tips

1. Use relative paths for portability:
```json
{
    "adapters": {
        "local_adapters": [
            {
                "name": "custom_adapter",
                "path": "./adapters/custom_adapter"
            }
        ]
    }
}
```

2. Mix local and hub adapters:
```json
{
    "adapters": {
        "hub_adapters": [
            {
                "name": "go_adapter",
                "repo_id": "your-username/go-programming-adapter"
            }
        ],
        "local_adapters": [
            {
                "name": "custom_adapter",
                "path": "./adapters/custom_adapter"
            }
        ]
    }
}
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 



