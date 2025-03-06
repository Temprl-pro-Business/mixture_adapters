from mixture_adapters import MixtureOfAdapters
import asyncio

async def main():
    system = MixtureOfAdapters(
        config_path="config.json",
        model_config_path="model_config.json",
        verbose=True,
        api_server=True,
        api_host="0.0.0.0",
        api_port=8000
    )
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        system.stop_api_server()

if __name__ == "__main__":
    asyncio.run(main())