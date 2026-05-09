"""FastAPI application entrypoint for SHL recommender service."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import SHLAgent
from catalog import CatalogManager
from models import ChatRequest, ChatResponse, HealthResponse

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

catalog_manager: CatalogManager | None = None
agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize catalog and agent on startup using non-blocking executor."""
    global catalog_manager, agent
    catalog_manager = CatalogManager()
    Path("./catalog_data").mkdir(parents=True, exist_ok=True)
    logger.info("Starting catalog initialization.")
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, catalog_manager.initialize)
    agent = SHLAgent(catalog_manager)
    logger.info("Service ready.")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Always return readiness-compatible health status."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process one stateless chat request with safe timeout and fallback response."""
    if agent is None:
        return ChatResponse(
            reply="Service is still starting. Please retry in a few seconds.",
            recommendations=[],
            end_of_conversation=False,
        )
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(agent.reply, request.messages), timeout=28
        )
        return response
    except TimeoutError:
        logger.error("Chat request timed out.", exc_info=True)
        return ChatResponse(
            reply="I could not finish within the time limit. Please resend your request in a shorter form.",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as exc:
        logger.error("Endpoint error: %s", exc, exc_info=True)
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
