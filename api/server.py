"""
Module for MixtureOfAdapters server.
"""

import json
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

class GenerateRequest(BaseModel):
    """Request for text generation."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    stream: bool = True
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    
class GenerateResponse(BaseModel):
    """Response from text generation."""
    content: str
    model: str
    
class APIServer:
    """MixtureOfAdapters server."""
    
    def __init__(self, mixture_adapters, host: str = "0.0.0.0", port: int = 8000):
        self.mixture_adapters = mixture_adapters
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Mixture of Adapters API",
            description="API for Mixture of Adapters",
            version="1.0.0"
        )
        self._setup_routes()
        self._setup_middleware()
        
    def _setup_middleware(self):
        """Setup CORS middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/generate")
        async def generate(request: GenerateRequest):
            """Generate text."""
            try:
                query = next(
                    (msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"),
                    ""
                )
                
                if request.stream:
                    return EventSourceResponse(
                        self._generate_stream(query, request.messages),
                        media_type="text/event-stream"
                    )
                
                # Generate complete response
                response_text = ""
                async for chunk in self.mixture_adapters.generate_response(
                    query=query,
                    messages=request.messages,
                    stream=False
                ):
                    response_text += chunk
                
                return GenerateResponse(
                    content=response_text,
                    model=self.mixture_adapters.get_current_adapter() or "base"
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/models")
        async def list_models():
            """List available models."""
            models = []
            
            # Add base model
            base_model = self.mixture_adapters.model_config["model_settings"]["base_model"]["name"]
            models.append({
                "name": base_model,
                "type": "base"
            })
            
            # Add adapters
            for adapter in self.mixture_adapters.adapter_config.hub_adapters:
                models.append({
                    "name": adapter.name,
                    "type": "adapter",
                    "source": "hub"
                })
                
            for adapter in self.mixture_adapters.adapter_config.local_adapters:
                models.append({
                    "name": adapter.name,
                    "type": "adapter",
                    "source": "local"
                })
                
            return {"models": models}
            
    async def _generate_stream(
        self,
        query: str,
        messages: List[Dict[str, str]]
    ):
        """Generate streaming response."""
        try:
            async for chunk in self.mixture_adapters.generate_response(
                query=query,
                messages=messages,
                stream=True
            ):
                if chunk:
                    yield self._format_chunk(chunk)
                    
        except Exception as e:
            print(f"Error in stream generation: {str(e)}")
            yield self._format_chunk(f"Error: {str(e)}")
            
    def _format_chunk(self, content: str) -> str:
        """Format a chunk for streaming."""
        data = {
            "content": content,
            "model": self.mixture_adapters.get_current_adapter() or "base",
            "timestamp": int(time.time())
        }
        return f"data: {json.dumps(data)}\n\n"
        
    def start(self):
        """Start the API server."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()