from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

try:  # pragma: no cover
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None
try:  # pragma: no cover
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SequenceScore:
    text: str
    total_nll: float
    token_count: int

    @property
    def total_logprob(self) -> float:
        return -self.total_nll

    def as_dict(self):
        return {
            "text": self.text,
            "total_nll": self.total_nll,
            "total_logprob": self.total_logprob,
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
    if device.type == "mps":
        # MPS benefits from reduced precision; float32 disables autocast
        # and is substantially slower for local Apple GPU inference.
        return torch.float16
    return torch.float32


def _tokenizer_help_error(model_name: str) -> RuntimeError:
    msg = (
        f"Failed to load tokenizer for {model_name!r}. "
        "If this is a SentencePiece/LLama/Mistral tokenizer, install "
        "`sentencepiece` and `protobuf` in your environment. "
        "For newer multimodal checkpoints, you may also need a recent "
        "`transformers` build and processor support."
    )
    return RuntimeError(msg)


def _last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank 2")
    flipped = attention_mask.to(dtype=torch.int64).flip(dims=[1])
    from_right = flipped.argmax(dim=1)
    return attention_mask.shape[1] - 1 - from_right


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Assign real tokens contiguous decoder positions, independent of padding."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank 2")
    position_ids = attention_mask.long().cumsum(-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


class LlamaNLLScorer:
    """
    Fast sentence-level NLL scorer for decoder-only models (defaults to Llama 3 8B).
    Uses right padding, mask-derived position IDs, and autocast/inference_mode to
    keep GPU passes quick while preserving unpadded decoder token positions.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        device_map: Optional[str] = None,
        compile_model: bool = False,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        padding_side: str = "right",
        use_attention_mask_position_ids: bool = True,
        score_in_float32: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = _select_dtype(self.device, dtype)
        self.use_attention_mask_position_ids = use_attention_mask_position_ids
        self.score_in_float32 = score_in_float32
        tokenizer_source = tokenizer_name or model_name
        self.tokenizer = self._load_tokenizer(
            tokenizer_source=tokenizer_source,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
        # Some tokenizers lack an explicit pad token; reuse EOS/UNK or add one.
        added_pad_token = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                added_pad_token = True
        self.tokenizer.padding_side = padding_side
        model_kwargs = {"torch_dtype": self.dtype}
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        if device_map:
            model_kwargs["device_map"] = device_map
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        self.model = self._load_model(model_name=model_name, model_kwargs=model_kwargs)
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()
        if compile_model and torch.cuda.is_available() and hasattr(torch, "compile"):
            # compile only after sending to device; helps on CUDA with PyTorch 2+
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def _load_tokenizer(
        self,
        tokenizer_source: str,
        use_fast: bool,
        trust_remote_code: bool,
    ):
        tokenizer_kwargs = {"use_fast": use_fast}
        if trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        try:
            return AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        except Exception as exc:
            if use_fast:
                retry_kwargs = dict(tokenizer_kwargs)
                retry_kwargs["use_fast"] = False
                try:
                    return AutoTokenizer.from_pretrained(tokenizer_source, **retry_kwargs)
                except Exception:
                    pass
            if AutoProcessor is not None:
                processor_kwargs = {}
                if trust_remote_code:
                    processor_kwargs["trust_remote_code"] = True
                try:
                    processor = AutoProcessor.from_pretrained(tokenizer_source, **processor_kwargs)
                    tokenizer = getattr(processor, "tokenizer", None)
                    if tokenizer is not None:
                        return tokenizer
                except Exception:
                    pass
            raise _tokenizer_help_error(tokenizer_source) from exc

    def _load_model(self, model_name: str, model_kwargs: Dict[str, object]):
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            if AutoModelForImageTextToText is not None:
                try:
                    return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to load model for {model_name!r} as either a causal LM or image-text model."
            ) from exc

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
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.use_attention_mask_position_ids:
                model_inputs["position_ids"] = _position_ids_from_attention_mask(attention_mask)
            with torch.inference_mode():
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=use_amp,
                ):
                    outputs = self.model(**model_inputs)
            logits = outputs.logits
            if self.device.type == "mps":
                # Some MPS reductions can produce NaNs here even when the forward
                # pass itself is fine. Move post-forward scoring to CPU float32.
                logits = logits.float().cpu()
                shift_labels = input_ids[:, 1:].cpu()
                shift_mask = attention_mask[:, 1:].cpu()
            else:
                shift_labels = input_ids[:, 1:]
                shift_mask = attention_mask[:, 1:]
            if self.score_in_float32:
                logits = logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            shift_logprobs = logprobs[:, :-1, :]
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

    def score_next_token_probabilities(
        self,
        prompts: Sequence[str],
        token_id_groups: Dict[str, Sequence[int]],
        batch_size: int = 8,
        max_length: Optional[int] = 256,
        show_progress: bool = False,
    ) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        if not prompts:
            return results
        use_amp = self.device.type in {"cuda", "mps"}
        batches = list(_chunked(prompts, batch_size))
        if show_progress and tqdm is not None:
            batches = tqdm(batches, desc="Scoring", unit="batch")
        token_id_groups = {
            label: [int(tok_id) for tok_id in tok_ids]
            for label, tok_ids in token_id_groups.items()
            if tok_ids
        }
        if not token_id_groups:
            raise ValueError("token_id_groups must contain at least one non-empty token list")
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
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.use_attention_mask_position_ids:
                model_inputs["position_ids"] = _position_ids_from_attention_mask(attention_mask)
            with torch.inference_mode():
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=use_amp,
                ):
                    outputs = self.model(**model_inputs)
            last_indices = _last_nonpad_indices(attention_mask)
            batch_indices = torch.arange(input_ids.shape[0], device=self.device)
            next_logits = outputs.logits[batch_indices, last_indices, :]
            if self.device.type == "mps":
                next_logits = next_logits.float().cpu()
            elif self.score_in_float32:
                next_logits = next_logits.float()
            next_probs = torch.softmax(next_logits, dim=-1)
            for i in range(next_probs.shape[0]):
                row: Dict[str, float] = {}
                for label, tok_ids in token_id_groups.items():
                    prob = next_probs[i, tok_ids].sum()
                    row[label] = float(prob.item())
                results.append(row)
        return results
