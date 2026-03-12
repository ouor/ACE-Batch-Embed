from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


DEFAULT_TEMPLATE = (
    "{vocal_identity} sings over a {genre_style} track. "
    "It has a {tempo}{key_clause}{bpm_clause} and feels {mood}. "
    "The {main_instrument} plays {main_playing_style}{sub_instrument_clause}. "
    "The vocals are {vocal_style}. "
    "The production is {production_style}{spatial_clause}{ambience_clause}{structure_clause}{energy_clause}."
)


@dataclass
class PromptSample:
    values: Dict[str, str]
    prompt: str


@dataclass
class SunoRandomPromptGenerator:
    attributes: Dict[str, List[str]]
    clauses: Dict[str, str]
    template: str = DEFAULT_TEMPLATE
    rng: random.Random = field(default_factory=random.Random)

    # 선택 슬롯 포함 확률
    include_key_prob: float = 0.7
    include_bpm_prob: float = 0.65
    include_sub_instrument_prob: float = 0.75
    include_spatial_prob: float = 0.8
    include_ambience_prob: float = 0.45
    include_structure_prob: float = 0.55
    include_energy_prob: float = 0.6

    @classmethod
    def from_json(
        cls,
        attributes_path: str | Path,
        clauses_path: str | Path,
        template: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "SunoRandomPromptGenerator":
        attributes_path = Path(attributes_path)
        clauses_path = Path(clauses_path)

        with attributes_path.open("r", encoding="utf-8") as f:
            attributes = json.load(f)

        with clauses_path.open("r", encoding="utf-8") as f:
            clauses = json.load(f)

        rng = random.Random(seed)
        return cls(
            attributes=attributes,
            clauses=clauses,
            template=template or DEFAULT_TEMPLATE,
            rng=rng,
        )

    def _pick(self, key: str) -> str:
        values = self.attributes.get(key)
        if not values:
            raise KeyError(f"Missing or empty attribute list for key: {key}")
        return self.rng.choice(values)

    def _maybe_include(self, probability: float) -> bool:
        return self.rng.random() < probability

    def _render_optional_clause(
        self,
        clause_key: str,
        include: bool,
        values: Dict[str, str],
    ) -> str:
        if not include:
            return ""
        clause_template = self.clauses.get(clause_key, "")
        if not clause_template:
            return ""
        return clause_template.format(**values)

    def _sample_values(self) -> Dict[str, str]:
        # 필수 슬롯
        values: Dict[str, str] = {
            "vocal_identity": self._pick("vocal_identity"),
            "genre_style": self._pick("genre_style"),
            "tempo": self._pick("tempo"),
            "mood": self._pick("mood"),
            "main_instrument": self._pick("main_instrument"),
            "main_playing_style": self._pick("main_playing_style"),
            "vocal_style": self._pick("vocal_style"),
            "production_style": self._pick("production_style"),
        }

        # 선택 슬롯 원재료도 미리 뽑아둠
        values.update(
            {
                "key_mode": self._pick("key_mode"),
                "bpm": self._pick("bpm"),
                "sub_instrument": self._pick("sub_instrument"),
                "sub_role": self._pick("sub_role"),
                "spatial_effects": self._pick("spatial_effects"),
                "background_ambience": self._pick("background_ambience"),
                "song_structure": self._pick("song_structure"),
                "energy_curve": self._pick("energy_curve"),
            }
        )

        # 조건부 절 생성
        values["key_clause"] = self._render_optional_clause(
            "key_clause",
            self._maybe_include(self.include_key_prob),
            values,
        )
        values["bpm_clause"] = self._render_optional_clause(
            "bpm_clause",
            self._maybe_include(self.include_bpm_prob),
            values,
        )
        values["sub_instrument_clause"] = self._render_optional_clause(
            "sub_instrument_clause",
            self._maybe_include(self.include_sub_instrument_prob),
            values,
        )
        values["spatial_clause"] = self._render_optional_clause(
            "spatial_clause",
            self._maybe_include(self.include_spatial_prob),
            values,
        )
        values["ambience_clause"] = self._render_optional_clause(
            "ambience_clause",
            self._maybe_include(self.include_ambience_prob)
            and values["background_ambience"] != "no additional ambience",
            values,
        )
        values["structure_clause"] = self._render_optional_clause(
            "structure_clause",
            self._maybe_include(self.include_structure_prob),
            values,
        )
        values["energy_clause"] = self._render_optional_clause(
            "energy_clause",
            self._maybe_include(self.include_energy_prob),
            values,
        )

        return values

    def _postprocess_text(self, text: str) -> str:
        # 공백/구두점 정리
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        return text.strip()

    def _is_valid(self, values: Dict[str, str]) -> bool:
        """
        아주 기본적인 호환성 필터.
        필요하면 여기 룰을 계속 추가하면 됨.
        """
        genre = values["genre_style"].lower()
        vocal = values["vocal_identity"].lower()
        vocal_style = values["vocal_style"].lower()
        structure = values["song_structure"].lower()
        main_style = values["main_playing_style"].lower()

        # 예시 1: 합창인데 너무 속삭이는 스타일은 제외
        if "choir" in vocal and ("whispery" in vocal_style or "fragile" in vocal_style):
            return False

        # 예시 2: bossa nova에 과격한 구조는 제외
        if "bossa nova" in genre and "explosive chorus" in structure:
            return False

        # 예시 3: cinematic orchestral에 trap rhythm 주악장은 제외
        if "cinematic orchestral" in genre and "trap rhythm" in main_style:
            return False

        return True

    def generate(self, max_retries: int = 50) -> PromptSample:
        for _ in range(max_retries):
            values = self._sample_values()
            if not self._is_valid(values):
                continue

            prompt = self.template.format(**values)
            prompt = self._postprocess_text(prompt)
            return PromptSample(values=values, prompt=prompt)

        raise RuntimeError("Failed to generate a valid prompt after max_retries.")

    def generate_many(self, count: int) -> Iterator[PromptSample]:
        for _ in range(count):
            yield self.generate()


if __name__ == "__main__":
    # 예시 사용법
    generator = SunoRandomPromptGenerator.from_json(
        attributes_path="./assets/attributes.json",
        clauses_path="./assets/clauses.json",
        seed=42,
    )

    for i, sample in enumerate(generator.generate_many(5), start=1):
        print(f"[{i}]")
        print(sample.prompt)
        # print(json.dumps(sample.values, ensure_ascii=False, indent=2))
        print("-" * 80)