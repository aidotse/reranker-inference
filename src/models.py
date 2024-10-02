from pydantic import BaseModel, Field


class PredictRequestModel(BaseModel):
    pairs: list[tuple[str, str]] = Field(
        description="Array of pairs of context and query"
    )


class PredictResponseModel(BaseModel):
    similarities: list[float]
