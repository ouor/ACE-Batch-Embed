"""Unit tests for text2music orchestration module."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from acestep.inference import (
    CreateSampleResult,
    GenerationConfig,
    GenerationParams,
    GenerationResult,
)
from acestep.text2music import Text2MusicOptions, run_text2music_generation


class TestRunText2MusicGeneration(unittest.TestCase):
    """Verify focused text2music orchestration behavior."""

    @patch("acestep.text2music.generate_music")
    @patch("acestep.text2music.create_sample")
    def test_sample_mode_success_path(self, mock_create_sample, mock_generate_music):
        """sample_mode should auto-fill params and continue to generation."""
        mock_create_sample.return_value = CreateSampleResult(
            caption="new caption",
            lyrics="new lyrics",
            bpm=110,
            duration=42.0,
            keyscale="C Major",
            language="en",
            timesignature="4/4",
            instrumental=False,
            success=True,
            status_message="ok",
        )
        mock_generate_music.return_value = GenerationResult(success=True)

        params = GenerationParams(
            task_type="text2music",
            caption="",
            lyrics="",
            duration=-1.0,
            vocal_language="unknown",
        )
        config = GenerationConfig(batch_size=1)
        llm_handler = SimpleNamespace(llm_initialized=True)

        result = run_text2music_generation(
            dit_handler=SimpleNamespace(),
            llm_handler=llm_handler,
            params=params,
            config=config,
            options=Text2MusicOptions(sample_mode=True, sample_query="pop ballad"),
        )

        self.assertTrue(result.success)
        self.assertEqual(params.caption, "new caption")
        self.assertEqual(params.lyrics, "new lyrics")
        self.assertEqual(params.bpm, 110)
        self.assertEqual(params.duration, 42.0)
        self.assertFalse(params.use_cot_metas)
        mock_create_sample.assert_called_once()
        mock_generate_music.assert_called_once()

    @patch("acestep.text2music.generate_music")
    @patch("acestep.text2music.create_sample")
    def test_auto_lyrics_failure_falls_back_to_instrumental(
        self,
        mock_create_sample,
        mock_generate_music,
    ):
        """Failed auto-lyrics generation should fall back to instrumental mode."""
        mock_create_sample.return_value = CreateSampleResult(
            success=False,
            error="lm unavailable",
            status_message="failed",
        )
        mock_generate_music.return_value = GenerationResult(success=True)

        params = GenerationParams(
            task_type="text2music",
            caption="sad piano",
            lyrics="",
            use_cot_lyrics=True,
        )
        config = GenerationConfig(batch_size=1)
        llm_handler = SimpleNamespace(llm_initialized=True)

        run_text2music_generation(
            dit_handler=SimpleNamespace(),
            llm_handler=llm_handler,
            params=params,
            config=config,
            options=Text2MusicOptions(),
        )

        self.assertEqual(params.lyrics, "[Instrumental]")
        self.assertTrue(params.instrumental)
        self.assertFalse(params.use_cot_lyrics)
        mock_create_sample.assert_called_once()
        mock_generate_music.assert_called_once()

    @patch("acestep.text2music.generate_music")
    @patch("acestep.text2music.format_sample")
    @patch("acestep.text2music.create_sample")
    def test_no_lm_preprocessing_when_disabled(
        self,
        mock_create_sample,
        mock_format_sample,
        mock_generate_music,
    ):
        """No text2music LM preprocessing should run when all related flags are disabled."""
        mock_generate_music.return_value = GenerationResult(success=True)

        params = GenerationParams(
            task_type="text2music",
            caption="house groove",
            lyrics="[Instrumental]",
            use_cot_lyrics=False,
        )
        config = GenerationConfig(batch_size=1)
        llm_handler = SimpleNamespace(llm_initialized=False)

        result = run_text2music_generation(
            dit_handler=SimpleNamespace(),
            llm_handler=llm_handler,
            params=params,
            config=config,
            options=Text2MusicOptions(sample_mode=False, use_format=False),
        )

        self.assertTrue(result.success)
        mock_create_sample.assert_not_called()
        mock_format_sample.assert_not_called()
        mock_generate_music.assert_called_once()


if __name__ == "__main__":
    unittest.main()
