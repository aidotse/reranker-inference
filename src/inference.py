import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CACHE_DIR = "/app/hf_cache"


logger = logging.getLogger(__name__)


class SimilarityClassifierModel:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded model {model_name} on device {self.model.device}")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1024,
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores
