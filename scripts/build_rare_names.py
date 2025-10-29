import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from ucimlrepo import fetch_ucirepo


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        renamed[col] = col.strip().lower().replace(" ", "_")
    return df.rename(columns=renamed)


def _find_column(columns: Sequence[str], candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    lowered = [c.lower() for c in columns]
    for want in candidates:
        want_lower = want.lower()
        for idx, col in enumerate(lowered):
            if want_lower in col:
                return columns[idx]
    if required:
        raise ValueError(f"Could not find column matching any of {candidates!r}. Available: {list(columns)!r}")
    return None


def _closest_years(available: Sequence[int], desired: Iterable[int]) -> List[int]:
    pool: Set[int] = set()
    if not available:
        return []
    for target in desired:
        if target in available:
            pool.add(target)
            continue
        above = [y for y in available if y >= target]
        below = [y for y in available if y < target]
        candidate: Optional[int] = None
        if above:
            candidate = above[0]
        elif below:
            candidate = below[-1]
        if candidate is not None:
            pool.add(candidate)
    return sorted(pool)


def _enforce_single_token(names: pd.Series) -> pd.Series:
    return names.str.replace("-", "", regex=False).where(
        names.str.match(r"^[A-Za-z]+$")
    )


def _assign_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure dataframe has prob_female/prob_male columns.
    Supports two formats:
      - Separate probability columns per gender.
      - A single probability column aligned with 'gender'.
    """
    cols = set(df.columns)
    if {"prob_female", "prob_male"}.issubset(cols):
        return df

    if "probability" not in cols:
        raise ValueError("Expected either prob_female/prob_male columns or a single 'probability' column.")
    if "gender" not in cols:
        raise ValueError("Single 'probability' column requires accompanying 'gender' column.")

    df["probability"] = df["probability"].astype(float)
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()

    def _prob_row(row):
        prob = float(row["probability"])
        if prob > 1.0:
            prob = prob / 100.0
        prob = max(0.0, min(prob, 1.0))
        gender = row["gender"]
        if gender.startswith("F"):
            if prob >= 0.5:
                prob_f = prob
            else:
                prob_f = 1.0 - prob
            prob_m = max(0.0, 1.0 - prob_f)
            return pd.Series({"prob_female": prob_f, "prob_male": prob_m})
        if gender.startswith("M"):
            if prob >= 0.5:
                prob_m = prob
            else:
                prob_m = 1.0 - prob
            prob_f = max(0.0, 1.0 - prob_m)
            return pd.Series({"prob_female": prob_f, "prob_male": prob_m})
        # Unknown gender, drop row by returning NaNs
        return pd.Series({"prob_female": float("nan"), "prob_male": float("nan")})

    probs = df.apply(_prob_row, axis=1)
    df = pd.concat([df, probs], axis=1)
    return df


def _stable_seed(base_seed: Optional[int], *parts: object) -> int:
    """
    Derive a deterministic seed for a composite key.
    Uses blake2b to avoid Python's randomized hash salt.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"::")
    derived = int.from_bytes(h.digest(), "big") & 0xFFFFFFFF
    base = 0 if base_seed is None else int(base_seed) & 0xFFFFFFFF
    return derived ^ base


def build_rare_names(
    min_year: int,
    years: Sequence[int],
    gender_prob_min: float,
    count_percentile: float,
    max_per_year_gender: Optional[int],
    limit_per_gender: Optional[int],
    source_label: str,
    seed: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    repo = fetch_ucirepo(id=591)
    features = repo.data.features.copy()
    targets = repo.data.targets
    if targets is not None and not targets.empty:
        features = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    features = _normalize_columns(features)
    cols = list(features.columns)

    name_col = _find_column(cols, ["name"])
    gender_col = _find_column(cols, ["gender", "sex"], required=False)
    year_col = _find_column(cols, ["year"], required=False)
    count_col = _find_column(cols, ["count", "frequency", "occurrence"], required=False)
    country_col = _find_column(cols, ["country", "region"], required=False)
    prob_f_col = _find_column(cols, ["probability_female", "prob_female", "female_probability"], required=False)
    prob_m_col = _find_column(cols, ["probability_male", "prob_male", "male_probability"], required=False)
    prob_col = None
    if prob_f_col is None or prob_m_col is None:
        prob_col = _find_column(cols, ["probability", "prob"], required=False)

    wanted_cols = [name_col]
    optional_mappings = []
    if gender_col:
        wanted_cols.append(gender_col)
    if year_col:
        wanted_cols.append(year_col)
    if count_col:
        wanted_cols.append(count_col)
    if country_col:
        wanted_cols.append(country_col)
    if prob_f_col:
        wanted_cols.append(prob_f_col)
    if prob_m_col:
        wanted_cols.append(prob_m_col)
    if prob_col and prob_col not in wanted_cols:
        wanted_cols.append(prob_col)

    df = features[wanted_cols].copy()
    rename_map = {name_col: "name"}
    if gender_col:
        rename_map[gender_col] = "gender"
    has_year = bool(year_col)
    if year_col:
        rename_map[year_col] = "year"
    if count_col:
        rename_map[count_col] = "count"
    if country_col:
        rename_map[country_col] = "country"
    if prob_f_col:
        rename_map[prob_f_col] = "prob_female"
    if prob_m_col:
        rename_map[prob_m_col] = "prob_male"
    if prob_col:
        rename_map[prob_col] = "probability"
    df = df.rename(columns=rename_map)

    if "gender" not in df.columns:
        raise ValueError("Dataset must provide a gender column to support gendered sampling.")

    if "count" not in df.columns:
        df["count"] = 1.0
    df["count"] = df["count"].astype(float)
    if "year" not in df.columns:
        df["year"] = min_year

    df["name"] = df["name"].astype(str).str.strip()
    df["name"] = _enforce_single_token(df["name"].str.title())
    df = df.dropna(subset=["name"])

    df = _assign_probabilities(df)

    df = df.dropna(subset=["prob_female", "prob_male"])
    df["prob_female"] = df["prob_female"].astype(float)
    df["prob_male"] = df["prob_male"].astype(float)

    if has_year:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["year"] = df["year"].fillna(min_year).astype(int)
        df = df[df["year"] >= min_year]
    else:
        df["year"] = min_year

    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["gender"] = df["gender"].map(lambda g: "F" if g.startswith("F") else ("M" if g.startswith("M") else None))
    df = df.dropna(subset=["gender"])

    if years and has_year:
        available = sorted(df["year"].unique())
        chosen_years = _closest_years(available, years)
        if not chosen_years:
            raise ValueError("No matching years found in dataset after applying filters.")
        df = df[df["year"].isin(chosen_years)]
    elif not year_col:
        # Dataset lacks year info; keep everything post min_year.
        chosen_years = []
    else:
        chosen_years = sorted(df["year"].unique())

    df["max_prob"] = df[["prob_female", "prob_male"]].max(axis=1)
    df = df[df["max_prob"] >= gender_prob_min]

    group_cols = ["gender", "year"] if has_year else ["gender"]

    groups = []
    for key, sub in df.groupby(group_cols):
        if sub.empty:
            continue
        thresh = sub["count"].quantile(count_percentile)
        if pd.isna(thresh):
            thresh = sub["count"].min()
        trimmed = sub[sub["count"] <= thresh].sort_values("count")
        if max_per_year_gender is not None and max_per_year_gender > 0 and len(trimmed) > max_per_year_gender:
            # Derive a stable seed per group to produce deterministic sampling.
            if isinstance(key, tuple):
                seed_parts = key
            else:
                seed_parts = (key,)
            group_seed = _stable_seed(seed, *seed_parts)
            trimmed = trimmed.sample(n=max_per_year_gender, random_state=group_seed, replace=False)
        groups.append(trimmed)

    if not groups:
        raise ValueError("No rare names found under the current filtering strategy.")

    rare = pd.concat(groups, axis=0)
    rare = rare.sort_values(["gender", "count", "max_prob", "name"], ascending=[True, True, False, True])
    rare = rare.drop_duplicates(subset=["name", "gender"], keep="first")

    if limit_per_gender is not None and limit_per_gender > 0:
        sampled = []
        for gender, sub in rare.groupby("gender"):
            limit = min(limit_per_gender, len(sub))
            if limit <= 0:
                continue
            gender_seed = _stable_seed(seed, gender, "final")
            selection = sub.sample(n=limit, random_state=gender_seed, replace=False)
            sampled.append(selection)
        if sampled:
            rare = pd.concat(sampled, axis=0)
    rare = rare.sort_values(["gender", "name"]).reset_index(drop=True)

    rare["source"] = source_label
    rare_cols = ["name", "gender"]
    if has_year:
        rare_cols.append("year")
    rare_cols.extend(["prob_female", "prob_male", "source", "max_prob"])
    rare = rare[rare_cols]
    rare = rare.reset_index(drop=True)

    filtered = df.reset_index(drop=True)
    if not has_year and "year" in filtered.columns:
        filtered = filtered.drop(columns=["year"])
    if "country" in filtered.columns:
        filtered = filtered.drop(columns=["country"])

    return rare, filtered, has_year


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate gender probabilities per name using availability-weighted counts.
    This guards against low-count outliers that would otherwise flip the
    perceived gender of common names (e.g., Todd -> F).
    """
    working = df.copy()
    if "count" not in working.columns:
        working["count"] = 1.0
    working["count"] = working["count"].fillna(1.0).astype(float).clip(lower=0.0)

    for col in ("prob_female", "prob_male"):
        working[col] = working[col].fillna(0.0).astype(float).clip(lower=0.0, upper=1.0)

    working["female_weight"] = working["prob_female"] * working["count"]
    working["male_weight"] = working["prob_male"] * working["count"]

    grouped = working.groupby("name", sort=True).agg(
        female_weight=("female_weight", "sum"),
        male_weight=("male_weight", "sum"),
        total_count=("count", "sum"),
    )

    grouped["total_weight"] = grouped["female_weight"] + grouped["male_weight"]
    grouped = grouped[grouped["total_weight"] > 0]
    if grouped.empty:
        return pd.DataFrame(columns=["name", "gender", "prob_female", "prob_male"])

    grouped["prob_female"] = grouped["female_weight"] / grouped["total_weight"]
    grouped["prob_male"] = grouped["male_weight"] / grouped["total_weight"]
    grouped["gender"] = grouped.apply(
        lambda row: "F" if row["prob_female"] >= row["prob_male"] else "M",
        axis=1,
    )

    cols = ["gender", "prob_female", "prob_male"]
    lookup = grouped[cols].reset_index()
    lookup = lookup.sort_values(["gender", "name"]).reset_index(drop=True)
    return lookup


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a rare-name bank from the Gender by Name dataset.")
    ap.add_argument("--min-year", type=int, default=1930, help="Discard entries earlier than this year (default: 1930).")
    ap.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=[1930, 1970, 2010],
        help="Target years to sample from; the closest available year is used for each value.",
    )
    ap.add_argument(
        "--gender-prob-min",
        type=float,
        default=0.9,
        help="Minimum probability mass for the dominant gender (default: 0.9).",
    )
    ap.add_argument(
        "--count-percentile",
        type=float,
        default=0.05,
        help="Percentile threshold (per year/gender) for defining rarity (default: 0.05).",
    )
    ap.add_argument(
        "--max-per-year-gender",
        type=int,
        default=0,
        help="Optional cap on rare names per year/gender bucket (<=0 disables the cap; default: disabled).",
    )
    ap.add_argument(
        "--limit-per-gender",
        type=int,
        default=1000,
        help="Cap on total names retained per gender (default: 1000).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling (default: 0).",
    )
    ap.add_argument(
        "--rare-out",
        type=Path,
        default=Path("data/external/rare_names.tsv"),
        help="Output TSV for rare names (default: data/external/rare_names.tsv).",
    )
    ap.add_argument(
        "--lookup-out",
        type=Path,
        default=Path("data/external/name_gender_lookup.tsv"),
        help="Output TSV for gender lookup (default: data/external/name_gender_lookup.tsv).",
    )
    ap.add_argument(
        "--source-label",
        default="gender_by_name",
        help="Source label to include in the output TSV (default: gender_by_name).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rare, filtered, has_year = build_rare_names(
        min_year=args.min_year,
        years=args.years,
        gender_prob_min=args.gender_prob_min,
        count_percentile=args.count_percentile,
        max_per_year_gender=args.max_per_year_gender,
        limit_per_gender=args.limit_per_gender,
        source_label=args.source_label,
        seed=args.seed,
    )

    lookup = build_lookup(filtered)

    rare_path = args.rare_out
    lookup_path = args.lookup_out
    rare_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    rare_out_df = rare.drop(columns=["max_prob"])
    if not has_year and "year" in rare_out_df.columns:
        rare_out_df = rare_out_df.drop(columns=["year"])
    rare_out_df.to_csv(rare_path, sep="\t", index=False)

    if not has_year and "year" in lookup.columns:
        lookup = lookup.drop(columns=["year"])
    lookup.to_csv(lookup_path, sep="\t", index=False)

    print(f"Wrote {len(rare)} rare names to {rare_path}")
    print(f"Wrote {len(lookup)} lookup entries to {lookup_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(1)
