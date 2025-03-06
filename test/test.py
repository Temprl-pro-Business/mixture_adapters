from mixture_adapters.client import MixtureClient

def main():
    # Initialize client
    client = MixtureClient("http://localhost:8000")
    
    # # List available models
    # models = client.list_models()
    # print("Available models:", models)
    
    # Your query
    messages = [
        {"role": "user", "content": "文本纠错：\n少先队员因该为老人让坐。"}
    ]
    
    # Stream response
    print("\nGenerating response:\n")
    for chunk in client.generate(messages=messages, stream=True):
        print(chunk, end="", flush=True)
    print()  # Final newline
    
    # Or get complete response
    # response = client.generate(messages=messages, stream=False)
    # print(response["content"])

if __name__ == "__main__":
    main()