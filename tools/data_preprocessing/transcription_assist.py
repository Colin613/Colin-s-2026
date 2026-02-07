"""
Transcription assistance tools for preparing training data.

This module provides utilities for:
- Generating transcriptions using ASR models
- Validating transcription files
- Format conversion for different transcription formats
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from loguru import logger


class TranscriptionAssistant:
    """
    Helper class for managing transcriptions for voice cloning training.

    Fish Speech requires .lab files with transcriptions for each audio segment.
    This tool helps manage and validate these files.
    """

    def __init__(self):
        pass

    def validate_lab_file(self, lab_path: str | Path) -> bool:
        """
        Validate a .lab file for training data.

        Args:
            lab_path: Path to .lab file

        Returns:
            True if valid, False otherwise
        """
        lab_path = Path(lab_path)

        if not lab_path.exists():
            logger.warning(f"Lab file does not exist: {lab_path}")
            return False

        content = lab_path.read_text(encoding="utf-8").strip()

        if not content:
            logger.warning(f"Lab file is empty: {lab_path}")
            return False

        # Basic validation: should have some text
        if len(content) < 2:
            logger.warning(f"Lab file has too little content: {lab_path}")
            return False

        return True

    def validate_dataset(
        self,
        data_dir: str | Path,
        check_audio: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Validate a complete dataset directory.

        Checks that:
        - Each .wav file has a corresponding .lab file
        - Each .lab file has valid content
        - Optionally checks that audio files exist

        Args:
            data_dir: Path to data directory
            check_audio: Whether to check audio file existence

        Returns:
            Dictionary with validation results
        """
        data_dir = Path(data_dir)

        results = {
            "valid": [],
            "missing_lab": [],
            "missing_audio": [],
            "empty_lab": [],
            "invalid_lab": [],
        }

        for speaker_dir in data_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            # Find all .wav files
            wav_files = list(speaker_dir.glob("*.wav"))

            for wav_file in wav_files:
                lab_file = speaker_dir / f"{wav_file.stem}.lab"

                # Check for corresponding lab file
                if not lab_file.exists():
                    results["missing_lab"].append(str(lab_file))
                    continue

                # Validate lab content
                if not self.validate_lab_file(lab_file):
                    content = lab_file.read_text(encoding="utf-8").strip()
                    if not content:
                        results["empty_lab"].append(str(lab_file))
                    else:
                        results["invalid_lab"].append(str(lab_file))
                    continue

                results["valid"].append({
                    "audio": str(wav_file),
                    "lab": str(lab_file),
                    "speaker": speaker_dir.name,
                })

            # Check for orphaned lab files
            if check_audio:
                lab_files = list(speaker_dir.glob("*.lab"))
                for lab_file in lab_files:
                    wav_file = speaker_dir / f"{lab_file.stem}.wav"
                    if not wav_file.exists():
                        results["missing_audio"].append(str(wav_file))

        # Log summary
        total_valid = len(results["valid"])
        total_issues = (
            len(results["missing_lab"]) +
            len(results["missing_audio"]) +
            len(results["empty_lab"]) +
            len(results["invalid_lab"])
        )

        logger.info(f"Validation complete: {total_valid} valid, {total_issues} issues")

        if results["missing_lab"]:
            logger.warning(f"Missing .lab files: {len(results['missing_lab'])}")
        if results["missing_audio"]:
            logger.warning(f"Missing audio files: {len(results['missing_audio'])}")
        if results["empty_lab"]:
            logger.warning(f"Empty .lab files: {len(results['empty_lab'])}")

        return results

    def create_lab_from_text(
        self,
        audio_dir: str | Path,
        transcription: str,
        time_range: Optional[str] = None,
    ) -> Path:
        """
        Create a .lab file from text transcription.

        Args:
            audio_dir: Directory containing audio files
            transcription: Text transcription
            time_range: Optional time range for filename (e.g., "00.00-05.23")

        Returns:
            Path to created .lab file
        """
        audio_dir = Path(audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        if time_range:
            lab_path = audio_dir / f"{time_range}.lab"
        else:
            # Find existing wav files to determine naming
            wav_files = list(audio_dir.glob("*.wav"))
            if wav_files:
                # Use first wav file's stem
                lab_path = audio_dir / f"{wav_files[0].stem}.lab"
            else:
                lab_path = audio_dir / "transcription.lab"

        lab_path.write_text(transcription, encoding="utf-8")
        logger.info(f"Created .lab file: {lab_path}")

        return lab_path

    def batch_create_labs(
        self,
        data_dir: str | Path,
        transcriptions: Dict[str, str],
    ) -> List[Path]:
        """
        Batch create .lab files from a transcription dictionary.

        Args:
            data_dir: Path to data directory
            transcriptions: Dict mapping time_range to transcription text
                           e.g., {"00.00-05.23": "Hello world"}

        Returns:
            List of created .lab file paths
        """
        data_dir = Path(data_dir)
        created = []

        for speaker_dir in data_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            for time_range, text in transcriptions.items():
                lab_path = speaker_dir / f"{time_range}.lab"
                lab_path.write_text(text, encoding="utf-8")
                created.append(lab_path)

        logger.info(f"Created {len(created)} .lab files")
        return created

    def format_transcription(
        self,
        text: str,
        language: str = "ko",
        normalize: bool = True,
    ) -> str:
        """
        Format and normalize transcription text.

        Args:
            text: Raw transcription text
            language: Language code (ko for Korean, zh for Chinese)
            normalize: Whether to apply normalization

        Returns:
            Formatted transcription
        """
        # Basic cleanup
        text = text.strip()

        if normalize:
            # Remove extra whitespace
            text = " ".join(text.split())

            # Language-specific normalization
            if language == "ko":
                # Korean-specific normalization
                # (Add any Korean text processing here)
                pass
            elif language == "zh":
                # Chinese-specific normalization
                # (Add any Chinese text processing here)
                pass

        return text

    def export_dataset_manifest(
        self,
        data_dir: str | Path,
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """
        Export a manifest of all training data.

        Args:
            data_dir: Path to data directory
            output_path: Path to output manifest file
            format: Output format (json or csv)
        """
        data_dir = Path(data_dir)
        output_path = Path(output_path)

        validation = self.validate_dataset(data_dir)

        manifest = {
            "data_dir": str(data_dir),
            "num_samples": len(validation["valid"]),
            "samples": validation["valid"],
            "issues": {
                k: len(v) for k, v in validation.items()
                if k != "valid" and isinstance(v, list)
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["audio", "lab", "speaker"])
                for sample in validation["valid"]:
                    writer.writerow([
                        sample["audio"],
                        sample["lab"],
                        sample["speaker"],
                    ])

        logger.info(f"Exported manifest to: {output_path}")

    def get_statistics(
        self,
        data_dir: str | Path,
    ) -> Dict[str, Any]:
        """
        Get statistics about the training dataset.

        Args:
            data_dir: Path to data directory

        Returns:
            Dictionary with dataset statistics
        """
        data_dir = Path(data_dir)

        stats = {
            "num_speakers": 0,
            "num_segments": 0,
            "total_duration": 0.0,
            "speakers": {},
        }

        import librosa

        for speaker_dir in data_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            stats["num_speakers"] += 1
            speaker_stats = {
                "num_segments": 0,
                "duration": 0.0,
            }

            for wav_file in speaker_dir.glob("*.wav"):
                lab_file = speaker_dir / f"{wav_file.stem}.lab"

                # Only count if lab file exists and is valid
                if not lab_file.exists():
                    continue
                if not self.validate_lab_file(lab_file):
                    continue

                # Get audio duration
                try:
                    audio, sr = librosa.load(str(wav_file), sr=None)
                    duration = len(audio) / sr
                except Exception as e:
                    logger.warning(f"Failed to load {wav_file}: {e}")
                    continue

                speaker_stats["num_segments"] += 1
                speaker_stats["duration"] += duration
                stats["num_segments"] += 1
                stats["total_duration"] += duration

            stats["speakers"][speaker_dir.name] = speaker_stats

        logger.info(f"Dataset: {stats['num_speakers']} speakers, "
                   f"{stats['num_segments']} segments, "
                   f"{stats['total_duration']/3600:.2f} hours")

        return stats


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse_parser = argparse.ArgumentParser(
        description="Transcription assistance tools"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("data_dir", help="Path to data directory")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get dataset statistics")
    stats_parser.add_argument("data_dir", help="Path to data directory")

    # Manifest command
    manifest_parser = subparsers.add_parser("manifest", help="Export dataset manifest")
    manifest_parser.add_argument("data_dir", help="Path to data directory")
    manifest_parser.add_argument("output", help="Output manifest file")
    manifest_parser.add_argument("--format", default="json", choices=["json", "csv"])

    args = parser.parse_args()

    assistant = TranscriptionAssistant()

    if args.command == "validate":
        results = assistant.validate_dataset(args.data_dir)
        print(f"Valid: {len(results['valid'])}")
        print(f"Missing .lab: {len(results['missing_lab'])}")
        print(f"Empty .lab: {len(results['empty_lab'])}")

    elif args.command == "stats":
        stats = assistant.get_statistics(args.data_dir)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.command == "manifest":
        assistant.export_dataset_manifest(
            args.data_dir,
            args.output,
            format=args.format,
        )
