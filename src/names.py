import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class NameRecord:
    name: str
    gender: str
    prob_female: float
    prob_male: float

    @property
    def confidence(self) -> float:
        return max(self.prob_female, self.prob_male)


class NameBank:
    """
    Provides gender-aware sampling of rare names while guarding against
    ambiguous entries (e.g., near-equal gender probabilities).
    """

    def __init__(
        self,
        rare_path: Path,
        lookup_path: Path,
        *,
        min_lookup_confidence: float = 0.75,
    ) -> None:
        self._min_conf = float(min_lookup_confidence)
        self._rare_path = Path(rare_path)
        self._lookup_path = Path(lookup_path)
        self._by_gender: Dict[str, List[str]] = {"F": [], "M": []}
        self._lookup: Dict[str, NameRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._lookup_path.exists():
            raise FileNotFoundError(f"Name lookup file not found: {self._lookup_path}")
        if not self._rare_path.exists():
            raise FileNotFoundError(f"Rare names file not found: {self._rare_path}")

        with self._lookup_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {"name", "gender", "prob_female", "prob_male"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Lookup file {self._lookup_path} missing columns: {sorted(missing)}")
            for row in reader:
                name = (row["name"] or "").strip().lower()
                gender = (row["gender"] or "").strip().upper()
                if not name or gender not in {"F", "M"}:
                    continue
                try:
                    prob_f = float(row["prob_female"])
                    prob_m = float(row["prob_male"])
                except (TypeError, ValueError):
                    continue
                record = NameRecord(name=name, gender=gender, prob_female=prob_f, prob_male=prob_m)
                if record.confidence < self._min_conf:
                    continue
                # Prefer the entry with higher confidence if duplicates exist.
                if name in self._lookup and self._lookup[name].confidence >= record.confidence:
                    continue
                self._lookup[name] = record

        with self._rare_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            required = {"name", "gender"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Rare name file {self._rare_path} missing columns: {sorted(missing)}")
            for row in reader:
                name_raw = (row["name"] or "").strip().lower()
                gender = (row["gender"] or "").strip().upper()
                if not name_raw or gender not in self._by_gender:
                    continue
                if name_raw not in self._lookup:
                    # Exclude rare names without a high-confidence lookup entry.
                    continue
                self._by_gender[gender].append(name_raw)

        for gender, names in self._by_gender.items():
            deduped = []
            seen = set()
            for name in names:
                if name not in seen:
                    deduped.append(name)
                    seen.add(name)
            deduped.sort()
            self._by_gender[gender] = deduped

    def gender_for(self, name: str) -> Optional[str]:
        """
        Return 'F' or 'M' if the provided name has a high-confidence gender,
        otherwise None.
        """
        if not name:
            return None
        record = self._lookup.get(name.strip().lower())
        if record is None:
            return None
        return record.gender

    def sample(self, gender: str, rng: Optional[random.Random] = None) -> Optional[str]:
        """
        Sample a rare name of the requested gender (lowercase).
        Returns None if the gender pool is empty.
        """
        gender = gender.upper()
        pool = self._by_gender.get(gender)
        if not pool:
            return None
        if rng is None:
            rng = random
        return rng.choice(pool)

    def has_gendered_pool(self, gender: str) -> bool:
        gender = gender.upper()
        return bool(self._by_gender.get(gender))

    def known_name(self, name: str) -> bool:
        return self.gender_for(name) is not None

    def genders_supported(self) -> Iterable[str]:
        return [g for g, pool in self._by_gender.items() if pool]

