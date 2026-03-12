"""Text2music-only orchestration: validate, preprocess, and generate."""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from acestep.inference import (
    GenerationConfig,
    GenerationParams,
    GenerationResult,
    create_sample,
    format_sample,
    generate_music,
)


@dataclass
class Text2MusicOptions:
    """Runtime options that are specific to text2music orchestration."""

    sample_mode: bool = False
    sample_query: str = ""
    use_format: bool = False


def _ensure_lm_initialized(llm_handler, feature_name: str) -> None:
    """Raise a clear error if a feature requires an uninitialized LM handler."""
    if not getattr(llm_handler, "llm_initialized", False):
        raise ValueError(f"{feature_name} requires an initialized LM handler.")


def _build_user_metadata_for_format(params: GenerationParams) -> Optional[dict]:
    """Build optional metadata constraints used by ``format_sample``."""
    metadata: dict = {}
    if params.bpm is not None:
        metadata["bpm"] = params.bpm
    if params.duration is not None and float(params.duration) > 0:
        metadata["duration"] = float(params.duration)
    if params.keyscale:
        metadata["keyscale"] = params.keyscale
    if params.timesignature:
        metadata["timesignature"] = params.timesignature
    if params.vocal_language and params.vocal_language != "unknown":
        metadata["language"] = params.vocal_language
    return metadata or None


def prepare_text2music_params(
    llm_handler,
    params: GenerationParams,
    options: Optional[Text2MusicOptions] = None,
) -> GenerationParams:
    """Apply text2music validation and optional LM preprocessing in-place."""
    options = options or Text2MusicOptions()
    if params.task_type != "text2music":
        raise ValueError("prepare_text2music_params only supports task_type='text2music'.")

    sample_mode = options.sample_mode or bool(options.sample_query and options.sample_query.strip())
    if sample_mode and params.use_cot_lyrics:
        logger.info("sample_mode enabled; disabling use_cot_lyrics.")
        params.use_cot_lyrics = False

    if not params.caption and not params.lyrics and not sample_mode:
        raise ValueError("caption or lyrics is required for text2music.")
    if params.use_cot_lyrics and not params.caption:
        raise ValueError("use_cot_lyrics requires caption for lyric generation.")

    format_has_duration = False

    if sample_mode:
        _ensure_lm_initialized(llm_handler, "sample_mode")
        sample_query = options.sample_query.strip() if options.sample_query else "NO USER INPUT"
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=sample_query,
            instrumental=bool(params.instrumental),
            vocal_language=params.vocal_language
            if params.vocal_language not in {"", "unknown"}
            else None,
            temperature=params.lm_temperature,
            top_k=params.lm_top_k,
            top_p=params.lm_top_p,
        )
        if not sample_result.success:
            raise RuntimeError(
                f"create_sample failed: {sample_result.error or sample_result.status_message}"
            )
        params.caption = sample_result.caption
        params.lyrics = sample_result.lyrics
        params.instrumental = bool(sample_result.instrumental)
        if params.bpm is None:
            params.bpm = sample_result.bpm
        if not params.keyscale:
            params.keyscale = sample_result.keyscale
        if not params.timesignature:
            params.timesignature = sample_result.timesignature
        if params.duration <= 0 and sample_result.duration is not None:
            params.duration = sample_result.duration
        if params.vocal_language in {"", "unknown"}:
            params.vocal_language = sample_result.language or params.vocal_language

    if options.use_format and (params.caption or params.lyrics):
        _ensure_lm_initialized(llm_handler, "use_format")
        format_result = format_sample(
            llm_handler=llm_handler,
            caption=params.caption or "",
            lyrics=params.lyrics or "",
            user_metadata=_build_user_metadata_for_format(params),
            temperature=params.lm_temperature,
            top_k=params.lm_top_k,
            top_p=params.lm_top_p,
        )
        if not format_result.success:
            raise RuntimeError(
                f"format_sample failed: {format_result.error or format_result.status_message}"
            )
        params.caption = format_result.caption or params.caption
        params.lyrics = format_result.lyrics or params.lyrics
        if format_result.duration:
            params.duration = format_result.duration
            format_has_duration = True
        if format_result.bpm:
            params.bpm = format_result.bpm
        if format_result.keyscale:
            params.keyscale = format_result.keyscale
        if format_result.timesignature:
            params.timesignature = format_result.timesignature

    if params.use_cot_lyrics:
        _ensure_lm_initialized(llm_handler, "use_cot_lyrics")
        sample_result = create_sample(
            llm_handler=llm_handler,
            query=params.caption,
            instrumental=False,
            vocal_language=params.vocal_language if params.vocal_language != "unknown" else None,
            temperature=params.lm_temperature,
            top_k=params.lm_top_k,
            top_p=params.lm_top_p,
        )
        if sample_result.success:
            params.caption = sample_result.caption
            params.lyrics = sample_result.lyrics
            if params.bpm is None:
                params.bpm = sample_result.bpm
            if not params.keyscale:
                params.keyscale = sample_result.keyscale
            if not params.timesignature:
                params.timesignature = sample_result.timesignature
            if params.duration <= 0 and sample_result.duration is not None:
                params.duration = sample_result.duration
            if params.vocal_language == "unknown":
                params.vocal_language = sample_result.language
            params.use_cot_metas = False
            params.use_cot_caption = False
        else:
            logger.warning(
                "Automatic lyric generation failed ({}). Falling back to instrumental.",
                sample_result.error,
            )
            params.lyrics = "[Instrumental]"
            params.instrumental = True
        params.use_cot_lyrics = False

    if sample_mode or format_has_duration:
        params.use_cot_metas = False

    return params


def run_text2music_generation(
    dit_handler,
    llm_handler,
    params: GenerationParams,
    config: GenerationConfig,
    save_dir: str = "output",
    options: Optional[Text2MusicOptions] = None,
) -> GenerationResult:
    """Run a text2music job end-to-end with optional LM preprocessing."""
    prepared_params = prepare_text2music_params(
        llm_handler=llm_handler,
        params=params,
        options=options,
    )
    return generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=prepared_params,
        config=config,
        save_dir=save_dir,
    )
