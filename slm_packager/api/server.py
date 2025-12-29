from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json
import asyncio

from ..config.models import SLMConfig, GenerationParams
from ..config.loader import ConfigLoader
from ..runtime import get_runtime, BaseRuntime

app = FastAPI(title="SLM Packager API", version="0.1.0")

# Global runtime instance
runtime: Optional[BaseRuntime] = None
config: Optional[SLMConfig] = None

class GenerateRequest(BaseModel):
    prompt: str
    params: Optional[GenerationParams] = None

@app.on_event("startup")
async def startup_event():
    # In a real app, we might load config from env var or args
    # For now, we expect the runtime to be initialized externally or lazily
    pass

@app.on_event("shutdown")
async def shutdown_event():
    global runtime
    if runtime:
        runtime.unload()

@app.post("/load")
async def load_model(config_path: str = Body(..., embed=True)):
    global runtime, config
    try:
        config = ConfigLoader.load(config_path)
        if runtime:
            runtime.unload()
        runtime = get_runtime(config)
        runtime.load()
        return {"status": "success", "message": f"Loaded model {config.model.name}"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/generate")
async def generate(request: GenerateRequest):
    global runtime, config
    if not runtime or not runtime.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load first.")
    
    params = request.params or config.params
    
    try:
        if params.stream:
            return StreamingResponse(
                _stream_generator(runtime, request.prompt, params),
                media_type="text/event-stream"
            )
        else:
            output = runtime.generate(request.prompt, params)
            return {"text": output}
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail="Generation timeout exceeded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def _stream_generator(rt, prompt, params):
    try:
        for chunk in rt.generate(prompt, params):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0)  # Yield control
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/info")
async def info():
    global config
    if config:
        return config.model_dump()
    return {"status": "no model loaded"}

@app.get("/health")
async def health():
    return {"status": "ok"}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
