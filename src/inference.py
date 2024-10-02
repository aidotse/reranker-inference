import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CACHE_DIR = "/app/hf_cache"


class SimilarityClassifierModel:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )
        self.model.eval()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1024,
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores
