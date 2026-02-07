"""
Data preprocessing tools for Yanbian Korean voice cloning.

This module provides utilities for preparing training data from raw audio files.

Usage:
    from tools.data_preprocessing import AudioProcessor, AudioSegmenter, TranscriptionAssistant

    # Process audio files
    processor = AudioProcessor()
    processor.process_directory("raw_audio", "processed_audio")

    # Segment long audio
    segmenter = AudioSegmenter()
    segmenter.segment_long_audio("long_audio.wav", "segments")

    # Manage transcriptions
    assistant = TranscriptionAssistant()
    assistant.validate_dataset("data")
"""

from .audio_processor import (
    AudioProcessor,
    process_with_ffmpeg,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    TARGET_FORMAT,
)

from .segmenter import AudioSegmenter

from .transcription_assist import TranscriptionAssistant

__all__ = [
    "AudioProcessor",
    "AudioSegmenter",
    "TranscriptionAssistant",
    "process_with_ffmpeg",
    "TARGET_SAMPLE_RATE",
    "TARGET_CHANNELS",
    "TARGET_FORMAT",
]
