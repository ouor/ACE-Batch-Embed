"""Promptgen-driven text2music batch orchestration utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from acestep.inference import GenerationConfig, GenerationParams
from acestep.text2music import Text2MusicOptions, run_text2music_generation
from modules.promptgen import PromptSample, SunoRandomPromptGenerator


@dataclass
class BatchRuntimeConfig:
    """Runtime configuration for prompt-driven text2music batch generation."""

    prompt_batch_size: int
    duration: float
    bpm: Optional[int]
    keyscale: str
    timesignature: str
    vocal_language: str
    lyrics: str
    instrumental: bool
    inference_steps: int
    guidance_scale: float
    generation_seed: int
    audio_format: str
    output_dir: str


@dataclass
class BatchItemResult:
    """Serializable per-item summary for batch generation results."""

    index: int
    prompt: str
    success: bool
    status_message: str
    error: Optional[str]
    audios: list[dict]


def _sanitize_audios_for_json(audios: list[dict]) -> list[dict]:
    """Drop non-serializable runtime fields from audio outputs."""
    sanitized: list[dict] = []
    for audio in audios:
        item = dict(audio)
        item.pop("tensor", None)
        sanitized.append(item)
    return sanitized


def build_prompt_generator(
    attributes_path: Path,
    clauses_path: Path,
    seed: Optional[int],
) -> SunoRandomPromptGenerator:
    """Create a prompt generator backed by JSON attribute/clauses files."""
    return SunoRandomPromptGenerator.from_json(
        attributes_path=attributes_path,
        clauses_path=clauses_path,
        seed=seed,
    )


def generate_prompt_samples(
    generator: SunoRandomPromptGenerator,
    count: int,
) -> list[PromptSample]:
    """Generate and collect prompt samples."""
    return list(generator.generate_many(count))


def build_generation_params(prompt: str, config: BatchRuntimeConfig) -> GenerationParams:
    """Build text2music generation parameters from one generated prompt."""
    return GenerationParams(
        task_type="text2music",
        caption=prompt,
        lyrics=config.lyrics,
        instrumental=config.instrumental,
        vocal_language=config.vocal_language,
        bpm=config.bpm,
        keyscale=config.keyscale,
        timesignature=config.timesignature,
        duration=config.duration,
        inference_steps=config.inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.generation_seed,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_cot_lyrics=False,
    )


def build_generation_config(config: BatchRuntimeConfig) -> GenerationConfig:
    """Build shared generation configuration for all batch items."""
    return GenerationConfig(batch_size=1, use_random_seed=True, audio_format=config.audio_format)


def initialize_dit_handler(
    project_root: Path,
    model_name: str,
    device: str,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
) -> object:
    """Initialize and return an ``AceStepHandler`` ready for generation."""
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    use_flash_attention = handler.is_flash_attention_available(device)
    status, success = handler.initialize_service(
        project_root=str(project_root),
        config_path=model_name,
        device=device,
        use_flash_attention=use_flash_attention,
        offload_to_cpu=offload_to_cpu,
        offload_dit_to_cpu=offload_dit_to_cpu,
    )
    if not success:
        raise RuntimeError(f"DiT initialization failed: {status}")
    logger.info("DiT initialized: {}", status)
    return handler


def run_prompt_batch_generation(
    dit_handler: object,
    llm_handler: Optional[object],
    samples: list[PromptSample],
    config: BatchRuntimeConfig,
) -> list[BatchItemResult]:
    """Generate one music output per prompt sample and return structured results."""
    gen_config = build_generation_config(config)
    options = Text2MusicOptions(sample_mode=False, use_format=False)
    results: list[BatchItemResult] = []

    for idx, sample in enumerate(samples, start=1):
        params = build_generation_params(prompt=sample.prompt, config=config)
        try:
            result = run_text2music_generation(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                params=params,
                config=gen_config,
                save_dir=config.output_dir,
                options=options,
            )
            results.append(
                BatchItemResult(
                    index=idx,
                    prompt=sample.prompt,
                    success=result.success,
                    status_message=result.status_message,
                    error=result.error,
                    audios=result.audios,
                )
            )
        except Exception as exc:
            logger.exception("Generation failed for batch item {}", idx)
            results.append(
                BatchItemResult(
                    index=idx,
                    prompt=sample.prompt,
                    success=False,
                    status_message="exception",
                    error=str(exc),
                    audios=[],
                )
            )
    return results


def write_batch_report(report_path: Path, results: list[BatchItemResult]) -> None:
    """Write batch results to a JSON report file."""
    payload = []
    for item in results:
        result_dict = asdict(item)
        result_dict["audios"] = _sanitize_audios_for_json(item.audios)
        payload.append(result_dict)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
