from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncIterator
from contextlib import asynccontextmanager
import uvicorn
import json
import asyncio

from .. import __version__
from ..config.models import GenerationParams
from .manager import ModelManager, ModelBusyError

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.model_manager = ModelManager()
    yield
    # Shutdown
    if hasattr(app.state, "model_manager"):
        await app.state.model_manager.unload()

app = FastAPI(
    title="SLM Packager API",
    version=__version__,
    lifespan=lifespan
)

class GenerateRequest(BaseModel):
    prompt: str
    params: Optional[GenerationParams] = None

@app.post("/load")
async def load_model(request: Request, config_path: str = Body(..., embed=True)):
    manager: ModelManager = request.app.state.model_manager
    try:
        config = await manager.load(config_path)
        return {"status": "success", "message": f"Loaded model {config.model.name}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    manager: ModelManager = request.app.state.model_manager

    if not manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load first.")

    try:
        result = await manager.generate(body.prompt, body.params)

        if isinstance(result, str):
            return {"text": result}

        # Check if result is a generator/iterator (streaming) — exclude strings
        if hasattr(result, '__next__') or hasattr(result, '__aiter__'):
            return StreamingResponse(
                _stream_wrapper(result),
                media_type="text/event-stream"
            )

        return {"text": result}

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timeout exceeded")
    except ModelBusyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def _stream_wrapper(iterator) -> AsyncIterator[str]:
    """Wrap streaming iterators for SSE responses."""
    loop = asyncio.get_running_loop()
    try:
        if hasattr(iterator, "__aiter__"):
            async for chunk in iterator:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
            return

        while True:
            # Run blocking next() call in a thread to avoid blocking the event loop
            chunk = await loop.run_in_executor(None, lambda: next(iterator, None))
            if chunk is None:
                break
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except StopIteration:
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/info")
async def info(request: Request):
    manager: ModelManager = request.app.state.model_manager
    if manager.config:
        return manager.config.model_dump()
    return {"status": "no model loaded"}

@app.get("/health")
async def health():
    return {"status": "ok"}

def start_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
