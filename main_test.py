"""Unit tests for promptgen-driven batch generation orchestration in ``main.py``."""

import unittest
from unittest.mock import patch

from acestep.inference import GenerationResult
from modules.promptgen import PromptSample

from modules.prompt_batch_runner import (
    BatchRuntimeConfig,
    build_generation_params,
    generate_prompt_samples,
    run_prompt_batch_generation,
)


class _FakeGenerator:
    """Test double for ``SunoRandomPromptGenerator``."""

    def __init__(self, prompts):
        self._prompts = prompts

    def generate_many(self, count):
        for idx in range(count):
            yield PromptSample(values={"idx": str(idx)}, prompt=self._prompts[idx])


class TestMainOrchestration(unittest.TestCase):
    """Verify prompt-driven batch orchestration helpers."""

    def setUp(self):
        self.config = BatchRuntimeConfig(
            prompt_batch_size=2,
            duration=-1.0,
            bpm=None,
            keyscale="",
            timesignature="",
            vocal_language="unknown",
            lyrics="[Instrumental]",
            instrumental=True,
            inference_steps=8,
            guidance_scale=7.0,
            generation_seed=-1,
            audio_format="flac",
            output_dir="output",
        )

    def test_generate_prompt_samples_returns_requested_count(self):
        """Prompt collection should preserve count and ordering."""
        generator = _FakeGenerator(["p1", "p2"])
        samples = generate_prompt_samples(generator=generator, count=2)

        self.assertEqual(2, len(samples))
        self.assertEqual("p1", samples[0].prompt)
        self.assertEqual("p2", samples[1].prompt)

    def test_build_generation_params_disables_lm_cot_flags(self):
        """Prompt-based generation should default to direct caption usage."""
        params = build_generation_params(prompt="house groove", config=self.config)

        self.assertEqual("text2music", params.task_type)
        self.assertEqual("house groove", params.caption)
        self.assertFalse(params.thinking)
        self.assertFalse(params.use_cot_metas)
        self.assertFalse(params.use_cot_caption)
        self.assertFalse(params.use_cot_language)
        self.assertFalse(params.use_cot_lyrics)

    @patch("modules.prompt_batch_runner.run_text2music_generation")
    def test_run_prompt_batch_generation_aggregates_per_item_results(self, mock_run):
        """Batch runner should execute one text2music call per prompt and collect outputs."""
        mock_run.side_effect = [
            GenerationResult(
                success=True,
                status_message="ok",
                audios=[{"path": "output/a.flac"}],
            ),
            GenerationResult(
                success=False,
                status_message="failed",
                error="bad input",
                audios=[],
            ),
        ]
        samples = [
            PromptSample(values={}, prompt="prompt-a"),
            PromptSample(values={}, prompt="prompt-b"),
        ]

        results = run_prompt_batch_generation(
            dit_handler=object(),
            llm_handler=None,
            samples=samples,
            config=self.config,
        )

        self.assertEqual(2, len(results))
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertEqual("prompt-a", results[0].prompt)
        self.assertEqual("bad input", results[1].error)
        self.assertEqual(2, mock_run.call_count)


if __name__ == "__main__":
    unittest.main()
