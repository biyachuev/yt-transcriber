"""Tests for GigaAM backend integration points (mocked to avoid downloads)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.transcriber import Transcriber
from src.config import TranscribeOptions, settings


@patch("transformers.AutoModel.from_pretrained")
def test_gigaam_loads_model_via_hf(mock_from_pretrained, tmp_path, monkeypatch):
    """Ensure GigaAM uses HF repo with correct revision and cache dir."""
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model  # preserve identity after .to()

    # Redirect cache dir to a temp location to avoid polluting real cache.
    original_cache_dir = settings.CACHE_DIR
    settings.CACHE_DIR = tmp_path
    try:
        transcriber = Transcriber(method=TranscribeOptions.GIGAAM_E2E_RNNT)
        transcriber._load_model()
    finally:
        settings.CACHE_DIR = original_cache_dir

    mock_from_pretrained.assert_called_once()
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["revision"] == "e2e_rnnt"
    assert kwargs["trust_remote_code"] is True
    assert Path(kwargs["cache_dir"]).samefile(tmp_path / "gigaam")
    assert transcriber.model is mock_model


def test_gigaam_chunking_uses_vad_and_guardrail(monkeypatch):
    """Chunking should call VAD-driven split points and respect 25s guardrail."""
    transcriber = Transcriber(method=TranscribeOptions.GIGAAM_E2E_RNNT)
    fake_audio = Path("/fake/audio.mp3")

    monkeypatch.setattr(transcriber, "_get_audio_duration", lambda _: 40.0)

    vad_segments = [(0.0, 5.0), (10.0, 12.0)]
    calc_split_points = [(0.0, 18.0), (18.0, 36.0)]

    with patch.object(
        transcriber, "_find_speech_boundaries", return_value=vad_segments
    ) as mock_vad, patch.object(
        transcriber,
        "_calculate_split_points",
        return_value=calc_split_points,
    ) as mock_calc, patch.object(
        transcriber, "_create_chunks_from_points", return_value=["chunked"]
    ) as mock_create:
        result = transcriber._split_audio_for_gigaam(fake_audio, target_chunk_duration=20.0)

    assert result == ["chunked"]
    mock_vad.assert_called_once_with(fake_audio)
    mock_calc.assert_called_once_with(40.0, 20.0, vad_segments)
    mock_create.assert_called_once()
    args, kwargs = mock_create.call_args
    assert args[0] == fake_audio
    assert args[1] == calc_split_points
    assert kwargs.get("prefix") == "giga"


def test_gigaam_chunking_handles_unknown_duration(monkeypatch, tmp_path):
    """When duration is unknown, return the original path sentinel chunk."""
    transcriber = Transcriber(method=TranscribeOptions.GIGAAM_E2E_CTC)
    fake_audio = tmp_path / "audio.mp3"
    fake_audio.touch()

    monkeypatch.setattr(transcriber, "_get_audio_duration", lambda _: None)

    result = transcriber._split_audio_for_gigaam(fake_audio)
    assert result == [(fake_audio, 0.0, 0.0)]
