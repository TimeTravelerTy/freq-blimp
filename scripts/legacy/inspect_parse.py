import argparse
import sys
from typing import Iterable, List

import spacy


def load_model():
    """
    Load spaCy's small English model.

    If the model is missing, print a clear error message instead of crashing
    with a cryptic ImportError.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        msg = (
            "Could not load spaCy model 'en_core_web_sm'. "
            "Install it with: python -m spacy download en_core_web_sm\n"
            f"Original error: {exc}"
        )
        raise SystemExit(msg)


DEFAULT_SENTENCES = [
    "Cheryl is forgetting about herself thinking about Beverly.",
    "Tonya is thinking about herself listening to Andrew.",
    "Galileo listens to Elizabeth.",
    "Randolf hasn't fled from Kayla.",
    "The Clintons research that phenomenon.",
    "It's himself that Omar cures.",
    "It's herself who Naomi had questioned.",
]


def iter_sentences(args: argparse.Namespace) -> Iterable[str]:
    if args.sentence:
        yield from args.sentence
    elif sys.stdin.isatty():
        # Interactive run with no piped input: use built-in examples immediately.
        yield from DEFAULT_SENTENCES
    else:
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield line


def format_parse(doc) -> List[str]:
    lines = []
    for token in doc:
        children = ",".join(str(child.i) for child in token.children)
        lines.append(
            f"{token.i:2d} {token.text:15} {token.lemma_:15} "
            f"{token.pos_:5} {token.tag_:6} {token.dep_:12} "
            f"head={token.head.i:2d} children=[{children}]"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Print spaCy parse details for sentences."
    )
    parser.add_argument(
        "sentence",
        nargs="*",
        help="Sentence(s) to parse. If omitted, read from stdin (one per line).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="spaCy pipe batch size (default: 32).",
    )
    args = parser.parse_args()

    nlp = load_model()
    texts = list(iter_sentences(args))
    if not texts:
        sys.exit("No sentences provided.")

    for doc in nlp.pipe(texts, batch_size=args.batch_size):
        print("\n" + doc.text)
        for line in format_parse(doc):
            print(line)


if __name__ == "__main__":
    main()
