import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

CACHE_DIR = "/app/hf_cache"


logger = logging.getLogger(__name__)


class SimilarityClassifierLLM:
    def __init__(self, model_name: str, trust_remote_code: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )

        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.model.eval()

        logger.info(f"Loaded model {model_name} on device {self.model.device}")

    def get_inputs(self, pairs, max_length=8):
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)
        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with torch.no_grad():
            inputs = self.get_inputs(pairs).to(self.model.device)
            scores = (
                self.model(**inputs, return_dict=True)
                .logits[:, -1, self.yes_loc]
                .view(-1)
                .float()
            )
            return scores


class SimilarityClassifierModel:
    def __init__(self, model_name: str, trust_remote_code: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )

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
            ).to(self.model.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores
