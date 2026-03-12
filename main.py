"""Entry point for promptgen-based text2music batch generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from modules.prompt_batch_runner import (
    BatchRuntimeConfig,
    build_prompt_generator,
    generate_prompt_samples,
    initialize_dit_handler,
    run_prompt_batch_generation,
    write_batch_report,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for prompt-driven batch generation."""
    parser = argparse.ArgumentParser(description="Promptgen-based text2music batch runner")
    parser.add_argument("--attributes-path", type=Path, default=Path("assets/attributes.json"))
    parser.add_argument("--clauses-path", type=Path, default=Path("assets/clauses.json"))
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="acestep-v15-turbo")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--offload-dit-to-cpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("output/promptgen_batch_report.json"),
    )
    parser.add_argument("--duration", type=float, default=-1.0)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--keyscale", type=str, default="")
    parser.add_argument("--timesignature", type=str, default="")
    parser.add_argument("--vocal-language", type=str, default="unknown")
    parser.add_argument("--lyrics", type=str, default="[Instrumental]")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--generation-seed", type=int, default=-1)
    parser.add_argument(
        "--audio-format",
        type=str,
        choices=["mp3", "wav", "flac", "wav32", "opus", "aac"],
        default="flac",
    )
    parser.set_defaults(instrumental=True)
    parser.add_argument("--instrumental", dest="instrumental", action="store_true")
    parser.add_argument("--with-vocals", dest="instrumental", action="store_false")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> BatchRuntimeConfig:
    """Build strongly-typed runtime configuration from parsed CLI arguments."""
    return BatchRuntimeConfig(
        prompt_batch_size=args.prompt_batch_size,
        duration=args.duration,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        vocal_language=args.vocal_language,
        lyrics=args.lyrics,
        instrumental=args.instrumental,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        generation_seed=args.generation_seed,
        audio_format=args.audio_format,
        output_dir=args.output_dir,
    )


def main() -> None:
    """Run prompt generation and text2music batch generation end-to-end."""
    args = parse_args()
    config = build_runtime_config(args)
    project_root = Path(__file__).resolve().parent

    prompt_generator = build_prompt_generator(
        attributes_path=args.attributes_path,
        clauses_path=args.clauses_path,
        seed=args.prompt_seed,
    )
    samples = generate_prompt_samples(prompt_generator, config.prompt_batch_size)

    dit_handler = initialize_dit_handler(
        project_root=project_root,
        model_name=args.model_name,
        device=args.device,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=args.offload_dit_to_cpu,
    )
    results = run_prompt_batch_generation(
        dit_handler=dit_handler,
        llm_handler=None,
        samples=samples,
        config=config,
    )
    write_batch_report(args.report_path, results)
    logger.info("Batch generation completed. Report: {}", args.report_path)


if __name__ == "__main__":
    main()
