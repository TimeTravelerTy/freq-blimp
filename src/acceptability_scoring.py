from typing import Dict, Iterable, List, Sequence, Set, Tuple

from src.sentence_nll import LlamaNLLScorer, SequenceScore

IN_TEMPLATE_PROMPT = "The following sentence is grammatically acceptable.\n\n{sentence}"

YES_NO_SYSTEM_PROMPT = "Your task is to evaluate the quality of given text."
YES_NO_USER_PROMPT = (
    "Is the following sentence grammatically acceptable? "
    "Respond with Yes or No as your answer.\n\n{sentence}"
)
YES_NO_BASE_PROMPT = (
    "Your task is to evaluate the quality of given text.\n"
    "Is the following sentence grammatically acceptable?\n\n"
    "{sentence}\n"
    "Respond with Yes or No as your answer. Answer:"
)

SUPPORTED_METHODS = (
    "nll",
    "in_template_lp",
    "in_template_meanlp",
    "in_template_penlp",
    "yes_no",
    "ensemble",
)

METHOD_COMPONENTS = {
    "nll": {"plain"},
    "in_template_lp": {"template"},
    "in_template_meanlp": {"template"},
    "in_template_penlp": {"template"},
    "yes_no": {"yes_no"},
    "ensemble": {"template", "yes_no"},
}


def expand_required_components(methods: Iterable[str]) -> Set[str]:
    required: Set[str] = set()
    for method in methods:
        if method not in METHOD_COMPONENTS:
            raise ValueError(f"Unsupported method: {method}")
        required.update(METHOD_COMPONENTS[method])
    return required


def format_in_template(sentence: str) -> str:
    return IN_TEMPLATE_PROMPT.format(sentence=sentence)


def resolve_yes_no_prompt_format(scorer: LlamaNLLScorer, prompt_format: str) -> str:
    if prompt_format not in {"auto", "base", "chat"}:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")
    has_chat_template = bool(getattr(scorer.tokenizer, "chat_template", None))
    if prompt_format == "auto":
        return "chat" if has_chat_template else "base"
    if prompt_format == "chat" and not has_chat_template:
        raise ValueError("Tokenizer has no chat template, so chat prompt formatting is unavailable.")
    return prompt_format


def build_yes_no_prompts(
    scorer: LlamaNLLScorer,
    sentences: Sequence[str],
    prompt_format: str = "auto",
) -> Tuple[str, List[str]]:
    resolved_format = resolve_yes_no_prompt_format(scorer, prompt_format)
    prompts: List[str] = []
    if resolved_format == "chat":
        for sentence in sentences:
            messages = [
                {"role": "system", "content": YES_NO_SYSTEM_PROMPT},
                {"role": "user", "content": YES_NO_USER_PROMPT.format(sentence=sentence)},
            ]
            prompts.append(
                scorer.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    else:
        for sentence in sentences:
            prompts.append(YES_NO_BASE_PROMPT.format(sentence=sentence))
    return resolved_format, prompts


def _single_token_ids_for_variants(
    scorer: LlamaNLLScorer,
    variants: Sequence[str],
) -> List[int]:
    token_ids: List[int] = []
    for variant in variants:
        ids = scorer.tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(int(ids[0]))
    return sorted(set(token_ids))


def yes_no_token_groups(scorer: LlamaNLLScorer) -> Dict[str, List[int]]:
    groups = {
        "yes": _single_token_ids_for_variants(scorer, ["Yes", " Yes", "\nYes"]),
        "no": _single_token_ids_for_variants(scorer, ["No", " No", "\nNo"]),
    }
    if not groups["yes"] or not groups["no"]:
        raise ValueError(
            "Could not find single-token verbalizers for Yes/No with this tokenizer."
        )
    return groups


def sentence_nll_score(seq_score: SequenceScore) -> float:
    return -seq_score.total_nll


def template_lp_score(seq_score: SequenceScore) -> float:
    return seq_score.total_logprob


def template_meanlp_score(seq_score: SequenceScore) -> float:
    if seq_score.token_count <= 0:
        return float("-inf")
    return seq_score.total_logprob / seq_score.token_count


def template_penlp_score(seq_score: SequenceScore) -> float:
    if seq_score.token_count <= 0:
        return float("-inf")
    penalty = ((5.0 + seq_score.token_count) / 6.0) ** 0.8
    return seq_score.total_logprob / penalty


def score_yes_no_probabilities(
    scorer: LlamaNLLScorer,
    sentences: Sequence[str],
    batch_size: int,
    max_length: int,
    prompt_format: str = "auto",
    show_progress: bool = False,
) -> Tuple[str, List[Dict[str, float]]]:
    resolved_format, prompts = build_yes_no_prompts(
        scorer=scorer,
        sentences=sentences,
        prompt_format=prompt_format,
    )
    token_groups = yes_no_token_groups(scorer)
    next_token_probs = scorer.score_next_token_probabilities(
        prompts,
        token_groups,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=show_progress,
    )
    scores: List[Dict[str, float]] = []
    for row in next_token_probs:
        p_yes = row["yes"]
        p_no = row["no"]
        denom = p_yes + p_no
        score = p_yes / denom if denom > 0 else 0.5
        scores.append(
            {
                "score": score,
                "p_yes": p_yes,
                "p_no": p_no,
            }
        )
    return resolved_format, scores
