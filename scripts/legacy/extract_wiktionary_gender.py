#!/usr/bin/env python3
"""
Extract gendered English person nouns from a Wiktionary dump.

The script scans MediaWiki XML dumps (e.g., ``enwiktionary-latest-*``)
and collects titles that belong to the provided category names
(``Category:en:Women`` / ``Category:en:Men`` by default). To keep the
output compact, only simple ASCII alphabetic lemmas that contain an
English ``===Noun===`` section are persisted.

Usage:
    python scripts/extract_wiktionary_gender.py \
        --input data/external/enwiktionary-latest-pages-articles-multistream.xml \
        --output data/processed/wiktionary_gender_lemmas.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Pattern, Sequence, Set
import xml.etree.ElementTree as ET


CategoryMap = Mapping[str, str]

_CATEGORY_PATTERN = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)
_CATEGORY_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(c|cln)\s*\|([^}{]+)\}\}", re.IGNORECASE)
_LABEL_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(lb|lbl|label)\s*\|([^}{]+)\}\}", re.IGNORECASE)
_QUAL_TEMPLATE_PATTERN = re.compile(r"\{\{\s*(qualifier|qual|q|qq)\s*\|([^}{]+)\}\}", re.IGNORECASE)
_TOPICS_PATTERN = re.compile(r"\{\{\s*topics\s*\|([^}]+)\}\}", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"^[A-Za-z]+$")
_DEFAULT_CATEGORY_ALIASES = {
    "female": (
        "en:Female people",
        "en:Women",
        "en:Female family members",
        "en:Female animals",
    ),
    "male": (
        "en:Male people",
        "en:Men",
        "en:Male family members",
        "en:Male animals",
    ),
}
_BLOCK_CATEGORY_KEYWORDS = (
    "offensive",
    "derogatory",
    "vulgar",
    "slur",
    "sexual",
    "masturb",
    "lgbt",
    "lesbian",
    ":gay",
    " gay",
    "gay ",
    "gay-",
    "lgbtq",
    "bisexual",
    "queer",
    "transgender",
    "transsexual",
    "transvestite",
    "paedoph",
    "pedoph",
    "fetish",
    "porn",
)
_BLOCK_LABEL_KEYWORDS = (
    "offensive",
    "derogatory",
    "vulgar",
    "slur",
    "pejorative",
    "sexual",
    "masturbation",
    "lgbt",
    "lesbian",
    "gay",
    "lgbtq",
    "bisexual",
    "queer",
    "transgender",
    "transsexual",
    "transvestite",
    "paedoph",
    "pedoph",
    "fetish",
    "porn",
)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalize_category_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    if normalized.startswith(":"):
        normalized = normalized[1:]
    if normalized.lower().startswith("category:"):
        normalized = normalized.split(":", 1)[1]
        if not normalized:
            return None
        if not normalized.lower().startswith("en:"):
            normalized = f"en:{normalized}"
    elif not normalized.lower().startswith("en:"):
        normalized = f"en:{normalized}"
    return normalized


def _extract_categories(text: str) -> Sequence[str]:
    cats: list[str] = []
    if not text:
        return cats
    for match in _CATEGORY_PATTERN.findall(text):
        raw = match.split("|", 1)[0].strip()
        norm = _normalize_category_name(raw)
        if norm:
            cats.append(norm)
    for template_cat in _extract_template_categories(text):
        cats.append(template_cat)
    return cats


def _extract_template_categories(text: str) -> Sequence[str]:
    """
    Extract categories expressed via {{C|en|...}} templates.
    """
    results: list[str] = []
    if not text:
        return results
    for match in _CATEGORY_TEMPLATE_PATTERN.finditer(text):
        payload = match.group(2)
        parts = [segment.strip() for segment in payload.split("|")]
        if len(parts) < 2:
            continue
        lang = parts[0].lower()
        if lang != "en":
            continue
        for cat in parts[1:]:
            norm = _normalize_category_name(cat)
            if norm:
                results.append(norm)
    return results


def _normalize_lemma(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    stripped = title.strip()
    if not stripped or not stripped.isascii():
        return None
    if any(ch.isspace() for ch in stripped):
        return None
    lower = stripped.lower()
    if not _WORD_PATTERN.fullmatch(lower):
        return None
    return lower


def _page_text(elem: ET.Element, tag_name: str) -> Optional[str]:
    for child in elem:
        if _strip_ns(child.tag) == tag_name:
            return child.text
    return None


def _extract_text_from_revision(elem: ET.Element) -> Optional[str]:
    for child in elem:
        if _strip_ns(child.tag) == "text":
            return child.text
    return None


def iter_pages(xml_path: Path) -> Iterator[Dict[str, Optional[str]]]:
    """
    Stream individual <page> entries from a MediaWiki dump.
    """
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if _strip_ns(elem.tag) != "page":
            continue
        title = _page_text(elem, "title") or ""
        ns_text = _page_text(elem, "ns") or "0"
        try:
            ns = int(ns_text)
        except ValueError:
            ns = 0
        redirect = any(_strip_ns(child.tag) == "redirect" for child in elem)
        text = None
        for child in elem:
            if _strip_ns(child.tag) == "revision":
                text = _extract_text_from_revision(child)
        yield {
            "title": title,
            "ns": ns,
            "redirect": redirect,
            "text": text or "",
        }
        elem.clear()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract gendered lemma lists from a Wiktionary dump.")
    parser.add_argument("--input", required=True, help="Path to the enwiktionary XML dump.")
    parser.add_argument("--output", required=True, help="Destination JSON file for the compact lexicon.")
    parser.add_argument(
        "--category",
        action="append",
        metavar="GENDER=Category:en:Women",
        help="Mapping from gender label to Wiktionary category (repeatable). Defaults to female=en:Women, male=en:Men.",
    )
    parser.add_argument(
        "--topic-keyword",
        action="append",
        metavar="GENDER=keyword",
        help="Treat {{topics}} entries whose label matches the keyword as belonging to GENDER (repeatable).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum lemma length (default: 3).",
    )
    parser.add_argument(
        "--require-english-noun",
        dest="require_english_noun",
        action="store_true",
        default=True,
        help="Require '==English==' and '===Noun===' sections (default: enabled).",
    )
    parser.add_argument(
        "--no-require-english-noun",
        dest="require_english_noun",
        action="store_false",
        help="Disable noun-section filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on processed pages (useful for debugging).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250000,
        help="Print a progress line every N pages (default: 250k).",
    )
    return parser.parse_args(argv)


def _build_category_map(values: Optional[Iterable[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not values:
        for gender, categories in _DEFAULT_CATEGORY_ALIASES.items():
            for category in categories:
                mapping[category] = gender
        return mapping
    for entry in values:
        if "=" not in entry:
            raise ValueError(f"Invalid --category '{entry}'. Expected GENDER=Category:en:Women.")
        gender, category = entry.split("=", 1)
        gender = gender.strip().lower()
        category = category.strip()
        if not gender or not category:
            raise ValueError(f"Invalid --category '{entry}'.")
        if category.startswith("Category:"):
            category = category.split("Category:", 1)[1]
        if category.startswith(":"):
            category = category[1:]
        mapping[category] = gender
    if not mapping:
        raise ValueError("At least one --category mapping is required.")
    return mapping


def _build_topic_keyword_map(values: Optional[Iterable[str]]) -> tuple[Dict[str, Sequence[Pattern[str]]], Dict[str, Sequence[str]]]:
    defaults: MutableMapping[str, list[str]] = defaultdict(list)
    if not values:
        defaults["female"].extend(["female", "women", "woman"])
        defaults["male"].extend(["male", "men", "man"])
    else:
        for entry in values:
            if "=" not in entry:
                raise ValueError(f"Invalid --topic-keyword '{entry}'. Expected GENDER=keyword.")
            gender, keyword = entry.split("=", 1)
            gender = gender.strip().lower()
            keyword = keyword.strip().lower()
            if not gender or not keyword:
                raise ValueError(f"Invalid --topic-keyword '{entry}'.")
            defaults[gender].append(keyword)
    compiled: Dict[str, Sequence[Pattern[str]]] = {}
    summary: Dict[str, Sequence[str]] = {}
    for gender, keywords in defaults.items():
        patterns: list[Pattern[str]] = []
        normalized: list[str] = []
        for keyword in keywords:
            key = keyword.strip().lower()
            if not key:
                continue
            patterns.append(re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE))
            normalized.append(key)
        if patterns:
            compiled[gender] = tuple(patterns)
            summary[gender] = tuple(normalized)
    return compiled, summary


def _topic_gender_hits(text: str, topic_keywords: Mapping[str, Sequence[Pattern[str]]]) -> Sequence[str]:
    matches: list[str] = []
    if not text or not topic_keywords:
        return matches
    for match in _TOPICS_PATTERN.finditer(text):
        payload = match.group(1)
        parts = [segment.strip() for segment in payload.split("|") if segment.strip()]
        if len(parts) < 2:
            continue
        lang = parts[0].lower()
        if lang != "en":
            continue
        topics = parts[1:]
        for topic in topics:
            for gender, patterns in topic_keywords.items():
                if any(patt.search(topic) for patt in patterns):
                    matches.append(gender)
                    break
    return matches


def _has_blocked_category(categories: Sequence[str]) -> bool:
    for cat in categories:
        lower = cat.lower()
        for keyword in _BLOCK_CATEGORY_KEYWORDS:
            if keyword in lower:
                return True
    return False


def _has_blocked_label(text: Optional[str]) -> bool:
    if not text:
        return False

    def _check_parts(parts, skip_first=False):
        items = parts[1:] if skip_first and len(parts) > 1 else parts
        for item in items:
            token = item.strip().lower()
            if not token or token == "_":
                continue
            for keyword in _BLOCK_LABEL_KEYWORDS:
                if keyword in token:
                    return True
        return False

    for pattern, skip_first in (
        (_LABEL_TEMPLATE_PATTERN, True),
        (_QUAL_TEMPLATE_PATTERN, False),
    ):
        for match in pattern.finditer(text):
            payload = match.group(2)
            parts = [segment.strip() for segment in payload.split("|") if segment.strip()]
            if not parts:
                continue
            if skip_first:
                lang = parts[0].lower()
                if lang not in {"en", "eng", "en-us", "en-gb"}:
                    continue
            if _check_parts(parts, skip_first=skip_first):
                return True
    return False


def collect_gendered_lemmas(
    xml_path: Path,
    category_map: CategoryMap,
    *,
    min_length: int = 3,
    require_english_noun: bool = True,
    limit: Optional[int] = None,
    progress_every: int = 250000,
    topic_keywords: Optional[Mapping[str, Sequence[Pattern[str]]]] = None,
) -> Dict[str, Set[str]]:
    normalized_map = {cat: gender for cat, gender in category_map.items()}
    buckets: Dict[str, Set[str]] = defaultdict(set)
    processed = 0
    matched = 0
    for page in iter_pages(xml_path):
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"[wiktionary] processed={processed:,} matched={matched:,}", file=sys.stderr)
        if limit and processed > limit:
            break
        if page.get("ns") != 0 or page.get("redirect"):
            continue
        lemma = _normalize_lemma(page.get("title"))
        if not lemma:
            continue
        if len(lemma) < min_length:
            continue
        text = page.get("text") or ""
        if require_english_noun:
            if "==English==" not in text or "===Noun===" not in text:
                continue
        if _has_blocked_label(text):
            continue
        categories = _extract_categories(text)
        if _has_blocked_category(categories):
            continue
        genders = {normalized_map[cat] for cat in categories if cat in normalized_map}
        if topic_keywords:
            for gender in _topic_gender_hits(text, topic_keywords):
                genders.add(gender)
        if not genders:
            continue
        for gender in genders:
            buckets[gender].add(lemma)
        matched += 1
    print(f"[wiktionary] done processed={processed:,} pages matched={matched:,}", file=sys.stderr)
    return buckets


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"[error] dump not found: {xml_path}", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    category_map = _build_category_map(args.category)
    topic_keywords, topic_keyword_summary = _build_topic_keyword_map(args.topic_keyword)
    try:
        buckets = collect_gendered_lemmas(
            xml_path,
            category_map,
            min_length=args.min_length,
            require_english_noun=args.require_english_noun,
            limit=args.limit,
            progress_every=args.progress_every,
            topic_keywords=topic_keywords,
        )
    except KeyboardInterrupt:
        print("\n[warn] interrupted.", file=sys.stderr)
        return 2
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "dump": xml_path.name,
        "categories": category_map,
        "filters": {
            "min_length": args.min_length,
            "require_english_noun": args.require_english_noun,
            "ascii_only": True,
            "letters_only": True,
        },
        "topic_keywords": topic_keyword_summary,
        "counts": {gender: len(sorted(list(lemmas))) for gender, lemmas in buckets.items()},
        "lemmas": {gender: sorted(list(lemmas)) for gender, lemmas in buckets.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[wiktionary] wrote {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
