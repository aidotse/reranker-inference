import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from config import Config, get_log_config
from fastapi import FastAPI, Request
from inference import SimilarityClassifierModel
from models import PredictRequestModel, PredictResponseModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Setup and teardown events of the app"""
    # Setup
    config = Config()
    app.state.model = SimilarityClassifierModel(model_name=config.reranker_model_name)
    yield
    # Teardown


app = FastAPI(title="Reranker Inference", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponseModel)
def predict(request: Request, payload: PredictRequestModel) -> PredictResponseModel:
    scores = request.app.state.model.predict(payload.pairs)
    return PredictResponseModel(similarities=scores)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_config=get_log_config(),
        reload=bool(os.environ.get("DEV_MODE")),
    )
