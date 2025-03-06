"""
Module for handling chat completions using the language model.
"""

import asyncio
from threading import Thread
from typing import Dict, List, Optional, AsyncGenerator
from transformers import PreTrainedModel, PreTrainedTokenizer, TextIteratorStreamer

class ChatGenerator:
    """
    Handles chat completions using the language model and adapters.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """
        Initialize the chat generator.
        
        Args:
            model (PreTrainedModel): The language model to use
            tokenizer (PreTrainedTokenizer): The tokenizer to use
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response
            
        Yields:
            str: Generated text chunks if streaming, or complete response
        """
        try:
            # Apply chat template and tokenize
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer(
                [prompt],
                return_tensors="pt"
            ).input_ids.to(self.model.device)
            
            if stream:
                # Set up streaming with TextIteratorStreamer
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    timeout=None  # No timeout
                )
                
                # Set up generation kwargs
                generation_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "do_sample": self.do_sample,
                    "streamer": streamer,
                }
                
                # Start generation in a separate thread
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Stream the response
                async for text in self._async_iterate(streamer):
                    yield text
                    
            else:
                # Generate complete response
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                yield response[len(prompt):]
                
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            yield f"Error: {str(e)}"
            
    async def _async_iterate(self, streamer):
        """Convert synchronous iterator to async iterator."""
        for text in streamer:
            await asyncio.sleep(0)  # Allow other tasks to run
            yield text 