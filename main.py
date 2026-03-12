"""Entry point for promptgen-based text2music batch generation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from loguru import logger

from modules.embedding_batch_runner import (
    collect_encoded_tracks,
    embed_tracks,
    write_embedding_report,
)
from modules.prompt_batch_runner import (
    BatchRuntimeConfig,
    build_prompt_generator,
    generate_prompt_samples,
    initialize_dit_handler,
    run_prompt_batch_generation,
    write_batch_report,
)
from modules.sqlite_batch_runner import write_embeddings_to_timestamped_sqlite
from modules.storage_batch_runner import upload_embedded_tracks_to_storage
from modules.vectordb_batch_runner import upsert_embedded_tracks_to_qdrant


def load_project_dotenv(project_root: Path) -> None:
    """Load ``.env`` from project root when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def require_env(name: str) -> str:
    """Return required environment variable value or raise clear error."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


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
    parser.add_argument(
        "--embedding-report-path",
        type=Path,
        default=Path("output/promptgen_embedding_report.json"),
    )
    parser.add_argument("--embed-model-id", type=str, default="OpenMuQ/MuQ-MuLan-large")
    parser.add_argument("--embed-device", type=str, default="cuda")
    parser.set_defaults(enable_embedding=True)
    parser.add_argument("--enable-embedding", dest="enable_embedding", action="store_true")
    parser.add_argument("--skip-embedding", dest="enable_embedding", action="store_false")
    parser.set_defaults(enable_qdrant=True)
    parser.add_argument("--enable-qdrant", dest="enable_qdrant", action="store_true")
    parser.add_argument("--skip-qdrant", dest="enable_qdrant", action="store_false")
    parser.add_argument("--qdrant-collection", type=str, default="ace_step_tracks")
    parser.set_defaults(enable_upload=True)
    parser.add_argument("--enable-upload", dest="enable_upload", action="store_true")
    parser.add_argument("--skip-upload", dest="enable_upload", action="store_false")
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


def run_embedding_pipeline(args: argparse.Namespace, results: list) -> None:
    """Run embedding, upload, sqlite export, and optional Qdrant upsert."""
    encoded_tracks = collect_encoded_tracks(results)
    embedded_tracks = embed_tracks(
        tracks=encoded_tracks,
        model_id=args.embed_model_id,
        device=args.embed_device,
    )
    write_embedding_report(args.embedding_report_path, embedded_tracks)
    logger.info(
        "Embedding completed. Report: {} (tracks={})",
        args.embedding_report_path,
        len(embedded_tracks),
    )

    object_key_by_track_id: dict[str, str] | None = None
    if args.enable_upload and embedded_tracks:
        object_key_by_track_id = upload_embedded_tracks_to_storage(
            embedded_tracks=embedded_tracks,
            endpoint_url=require_env("S3_ENDPOINT_URL"),
            access_key_id=require_env("S3_ACCESS_KEY_ID"),
            secret_access_key=require_env("S3_SECRET_ACCESS_KEY"),
            bucket=require_env("S3_BUCKET"),
            region_name=os.environ.get("S3_REGION", "us-east-1"),
            key_prefix=os.environ.get("S3_KEY_PREFIX", "music"),
        )
        logger.info("Storage upload completed. Uploaded tracks={}", len(object_key_by_track_id))

    sqlite_path = write_embeddings_to_timestamped_sqlite(
        embedded_tracks=embedded_tracks,
        output_dir=args.output_dir,
        object_key_by_track_id=object_key_by_track_id,
    )
    logger.info("SQLite export completed. File: {}", sqlite_path)

    if args.enable_qdrant and embedded_tracks:
        upserted_count = upsert_embedded_tracks_to_qdrant(
            embedded_tracks=embedded_tracks,
            qdrant_url=require_env("QDRANT_URL"),
            qdrant_api_key=require_env("QDRANT_API_KEY"),
            collection_name=args.qdrant_collection,
            object_key_by_track_id=object_key_by_track_id,
        )
        logger.info(
            "Qdrant upsert completed. Collection: {} (tracks={})",
            args.qdrant_collection,
            upserted_count,
        )


def main() -> None:
    """Run prompt generation and text2music batch generation end-to-end."""
    args = parse_args()
    config = build_runtime_config(args)
    project_root = Path(__file__).resolve().parent
    load_project_dotenv(project_root)

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

    if args.enable_embedding:
        run_embedding_pipeline(args=args, results=results)


if __name__ == "__main__":
    main()
