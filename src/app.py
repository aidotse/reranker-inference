from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import uvicorn
from config import Config, get_log_config
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from inference import SimilarityClassifierModel
from models import PredictRequestModel, PredictResponseModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Setup and teardown events of the app"""
    # Setup
    config = Config()
    app.state.model = SimilarityClassifierModel(
        model_name=config.reranker_model_name,
        trust_remote_code=config.trust_remote_code,
    )
    app.state.api_key = config.api_key
    yield
    # Teardown


app = FastAPI(title="Reranker Inference", lifespan=lifespan)
security = HTTPBearer(auto_error=False)


def validate_credentials(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    api_key = request.app.state.api_key
    if api_key is None:
        return

    if credentials is not None and api_key == credentials.credentials:
        return

    raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid authorization header")


@app.post("/predict", response_model=PredictResponseModel)
def predict(
    request: Request,
    payload: PredictRequestModel,
    _=Depends(validate_credentials),
) -> PredictResponseModel:
    scores = request.app.state.model.predict(payload.pairs)
    return PredictResponseModel(similarities=scores)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_config=get_log_config(),
    )
