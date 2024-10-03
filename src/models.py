from pydantic import BaseModel, Field


class PredictRequestModel(BaseModel):
    pairs: list[tuple[str, str]] = Field(
        description="Array of pairs of context and query",
        examples=[[("Question", "Context")]],
    )


class PredictResponseModel(BaseModel):
    similarities: list[float] = Field(description="Similarity score logits")
