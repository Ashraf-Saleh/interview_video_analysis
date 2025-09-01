"""
FastAPI application entrypoint.
"""
import logging
from fastapi import FastAPI
from api.routes import router

logging.basicConfig(level=logging.DEBUG)
app = FastAPI(title="Interview Video Analysis API", version="1.0.0")
app.include_router(router)

@app.get("/health")
def health() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Simple status payload.
    """
    return {"status": "ok"}
