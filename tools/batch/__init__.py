"""
Batch processing tools for TTS dubbing workflows.

This module provides utilities for:
- Subtitle parsing and processing
- Batch TTS generation
- Task queue management
- Progress tracking
"""

from .subtitle_processor import (
    SubtitleProcessor,
    SubtitleEntry,
    CharacterVoice,
)

from .batch_processor import (
    BatchProcessor,
    BatchJobConfig,
    BatchJobStatus,
    mock_tts,
)

__all__ = [
    "SubtitleProcessor",
    "SubtitleEntry",
    "CharacterVoice",
    "BatchProcessor",
    "BatchJobConfig",
    "BatchJobStatus",
    "mock_tts",
]
