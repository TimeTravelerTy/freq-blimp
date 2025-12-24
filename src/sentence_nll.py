from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

try:  # pragma: no cover
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SequenceScore:
    text: str
    total_nll: float
    token_count: int

    def as_dict(self):
        return {
            "text": self.text,
            "total_nll": self.total_nll,
            "token_count": self.token_count,
        }


def _chunked(seq: Sequence[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _select_dtype(device: torch.device, requested: Optional[str]):
    if requested and requested != "auto":
        if requested == "bfloat16":
            return torch.bfloat16
        if requested == "float16":
            return torch.float16
        if requested == "float32":
            return torch.float32
    if device.type == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class LlamaNLLScorer:
    """
    Fast sentence-level NLL scorer for decoder-only models (defaults to Llama 3 8B).
    Uses left padding, autocast, and no_grad/inference_mode to keep GPU passes quick.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        device_map: Optional[str] = None,
        compile_model: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = _select_dtype(self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Llama tokenizers often lack an explicit pad token; reuse EOS for padding.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {"dtype": self.dtype}
        if device_map:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        if compile_model and torch.cuda.is_available() and hasattr(torch, "compile"):
            # compile only after sending to device; helps on CUDA with PyTorch 2+
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def score_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        max_length: Optional[int] = 256,
        show_progress: bool = False,
    ) -> List[SequenceScore]:
        results: List[SequenceScore] = []
        if not texts:
            return results
        # Enable autocast on CUDA/MPS; stay in full precision elsewhere.
        use_amp = self.device.type in {"cuda", "mps"}
        batches = list(_chunked(texts, batch_size))
        if show_progress and tqdm is not None:
            batches = tqdm(batches, desc="Scoring", unit="batch")
        for batch in batches:
            encoded = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
            with torch.inference_mode():
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=use_amp,
                ):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1)
            shift_logprobs = logprobs[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]
            nll = -shift_logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            nll = nll * shift_mask
            token_counts = shift_mask.sum(dim=1)
            total_nll = nll.sum(dim=1)
            for i, text in enumerate(batch):
                count = int(token_counts[i].item())
                results.append(
                    SequenceScore(
                        text=text,
                        total_nll=float(total_nll[i].item()),
                        token_count=count,
                    )
                )
        return results
