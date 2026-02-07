"""
Subtitle processing utilities for batch dubbing workflows.

This module provides tools for:
- SRT/ASS subtitle parsing
- Character identification and voice mapping
- Time-based text segmentation
- Subtitle alignment and optimization
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from loguru import logger


@dataclass
class SubtitleEntry:
    """A single subtitle entry with timing and text."""
    index: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    text: str
    speaker: Optional[str] = None  # Detected speaker/character
    voice_id: Optional[str] = None  # Mapped voice for TTS


@dataclass
class CharacterVoice:
    """Mapping between a character and their voice."""
    character: str
    voice_id: str
    aliases: List[str]  # Alternative names for the character


class SubtitleProcessor:
    """
    Process subtitle files for batch dubbing.

    Features:
    - SRT format parsing
    - Character/speaker detection
    - Voice mapping
    - Time alignment
    """

    # Time format patterns for SRT
    TIME_PATTERN = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def __init__(self):
        self.entries: List[SubtitleEntry] = []
        self.voice_mappings: Dict[str, CharacterVoice] = {}

    @staticmethod
    def parse_srt_time(time_str: str) -> float:
        """
        Convert SRT timestamp to seconds.

        Args:
            time_str: SRT timestamp like "00:01:23,456"

        Returns:
            Time in seconds (float)
        """
        match = re.match(r"(\d+):(\d+):(\d+)[,\.](\d+)", time_str)
        if match:
            hours, minutes, seconds, milliseconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return 0.0

    @staticmethod
    def format_srt_time(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format.

        Args:
            seconds: Time in seconds

        Returns:
            SRT timestamp string like "00:01:23,456"
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def parse_srt(self, srt_path: str | Path) -> List[SubtitleEntry]:
        """
        Parse an SRT subtitle file.

        Args:
            srt_path: Path to SRT file

        Returns:
            List of SubtitleEntry objects
        """
        srt_path = Path(srt_path)

        if not srt_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {srt_path}")

        content = srt_path.read_text(encoding="utf-8-sig")  # Handle BOM

        # Split by double newlines to get subtitle blocks
        blocks = re.split(r"\n\s*\n", content.strip())

        entries = []
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            # Parse index
            try:
                index = int(lines[0])
            except ValueError:
                continue

            # Parse timing
            time_match = self.TIME_PATTERN.search(lines[1])
            if not time_match:
                continue

            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
            start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

            # Parse text (may span multiple lines)
            text = "\n".join(lines[2:]).strip()

            # Remove HTML tags if present
            text = re.sub(r"<[^>]+>", "", text)

            # Remove SSA/ASS style tags
            text = re.sub(r"\{[^}]+\}", "", text)

            entry = SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
            entries.append(entry)

        self.entries = entries
        logger.info(f"Parsed {len(entries)} subtitle entries from {srt_path.name}")

        return entries

    def parse_ass(self, ass_path: str | Path) -> List[SubtitleEntry]:
        """
        Parse an ASS/SSA subtitle file.

        Args:
            ass_path: Path to ASS file

        Returns:
            List of SubtitleEntry objects
        """
        ass_path = Path(ass_path)

        if not ass_path_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {ass_path}")

        content = ass_path.read_text(encoding="utf-8-sig")

        entries = []

        # Parse [Events] section
        in_events = False
        format_line = ""

        for line in content.split("\n"):
            line = line.strip()

            if line == "[Events]":
                in_events = True
                continue

            if in_events and line.startswith("Format:"):
                format_line = line
                continue

            if in_events and line.startswith("Dialogue:"):
                # Parse dialogue line
                parts = line[9:].split(",", 9)  # Split into max 10 parts
                if len(parts) < 10:
                    continue

                # Extract timing (Start, End)
                start = self._parse_ass_time(parts[1].strip())
                end = self._parse_ass_time(parts[2].strip())

                # Extract text
                text = parts[9].strip()

                # Remove ASS tags
                text = re.sub(r"\{[^}]+\}", "", text)
                text = re.sub(r"\\N", "\n", text)  # Convert \N to newline
                text = text.strip()

                entry = SubtitleEntry(
                    index=len(entries) + 1,
                    start_time=start,
                    end_time=end,
                    text=text,
                )
                entries.append(entry)

        self.entries = entries
        logger.info(f"Parsed {len(entries)} subtitle entries from {ass_path.name}")

        return entries

    @staticmethod
    def _parse_ass_time(time_str: str) -> float:
        """Convert ASS timestamp to seconds."""
        match = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{2})", time_str)
        if match:
            h, m, s, cs = map(int, match.groups())
            return h * 3600 + m * 60 + s + cs / 100
        return 0.0

    def detect_characters(
        self,
        patterns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[int]]:
        """
        Detect characters/speakers from subtitle text.

        Uses common patterns like:
        - "Character: dialogue"
        - "[Character] dialogue"
        - "Character名: dialogue" (for Chinese/Japanese)

        Args:
            patterns: Custom regex patterns for character detection

        Returns:
            Dictionary mapping character names to list of entry indices
        """
        character_entries: Dict[str, List[int]] = {}

        # Default patterns for character detection
        default_patterns = {
            "colon_prefix": r"^([^:：]+?)[:：]\s*(.+)$",
            "bracket_prefix": r"^\[([^\]]+)\]\s*(.+)$",
            "parentheses_prefix": r"^【([^】]+)】\s*(.+)$",
        }

        # Combine with custom patterns
        all_patterns = {**default_patterns, **(patterns or {})}

        for i, entry in enumerate(self.entries):
            text = entry.text.strip()

            for pattern_name, pattern in all_patterns.items():
                match = re.match(pattern, text)
                if match:
                    character = match.group(1).strip()
                    dialogue = match.group(2).strip()

                    # Skip if character is too long (likely not a name)
                    if len(character) > 20:
                        continue

                    # Skip common non-character prefixes
                    if character.lower() in ["music", "sound", "fx", "sfx"]:
                        continue

                    if character not in character_entries:
                        character_entries[character] = []

                    character_entries[character].append(i)

                    # Update entry with detected speaker
                    entry.speaker = character
                    entry.text = dialogue

                    break

        logger.info(f"Detected {len(character_entries)} characters")
        for char, indices in character_entries.items():
            logger.debug(f"  {char}: {len(indices)} lines")

        return character_entries

    def set_voice_mappings(self, mappings: Dict[str, str]) -> None:
        """
        Set voice ID mappings for characters.

        Args:
            mappings: Dictionary mapping character names to voice IDs
        """
        for character, voice_id in mappings.items():
            self.voice_mappings[character] = CharacterVoice(
                character=character,
                voice_id=voice_id,
                aliases=[],
            )

        logger.info(f"Set {len(mappings)} voice mappings")

    def apply_voice_mappings(self) -> None:
        """Apply voice mappings to all subtitle entries."""
        for entry in self.entries:
            if entry.speaker and entry.speaker in self.voice_mappings:
                entry.voice_id = self.voice_mappings[entry.speaker].voice_id

        logger.info("Applied voice mappings to subtitle entries")

    def merge_short_entries(
        self,
        min_duration: float = 1.5,
        max_gap: float = 1.0,
        same_speaker_only: bool = True,
    ) -> None:
        """
        Merge short subtitle entries with adjacent entries.

        Args:
            min_duration: Minimum duration for entries (shorter will be merged)
            max_gap: Maximum gap between entries to merge
            same_speaker_only: Only merge if same speaker detected
        """
        new_entries = []
        i = 0

        while i < len(self.entries):
            current = self.entries[i]

            # Check if entry is too short
            if current.end_time - current.start_time >= min_duration:
                new_entries.append(current)
                i += 1
                continue

            # Look for next entry to merge with
            merged = False
            for j in range(i + 1, min(i + 5, len(self.entries))):
                next_entry = self.entries[j]

                # Check gap
                gap = next_entry.start_time - current.end_time
                if gap > max_gap:
                    break

                # Check speaker constraint
                if same_speaker_only and current.speaker != next_entry.speaker:
                    continue

                # Merge entries
                merged_entry = SubtitleEntry(
                    index=current.index,
                    start_time=current.start_time,
                    end_time=next_entry.end_time,
                    text=f"{current.text} {next_entry.text}",
                    speaker=current.speaker or next_entry.speaker,
                    voice_id=current.voice_id or next_entry.voice_id,
                )
                new_entries.append(merged_entry)
                i = j + 1
                merged = True
                break

            if not merged:
                new_entries.append(current)
                i += 1

        logger.info(f"Merged short entries: {len(self.entries)} -> {len(new_entries)}")
        self.entries = new_entries

    def optimize_punctuation(self) -> None:
        """Optimize punctuation for TTS (pause marks, sentence endings)."""
        for entry in self.entries:
            # Add pause at end if missing
            text = entry.text.strip()

            # Check if ending with punctuation
            if text and text[-1] not in {".", "!", "?", "。", "！", "？", "…"}:
                # Add ellipsis for natural pause
                entry.text = text + "…"

            # Ensure proper spacing after punctuation
            entry.text = re.sub(r"([.!?。！？])([^\s])", r"\1 \2", entry.text)

        logger.info("Optimized punctuation for TTS")

    def filter_by_duration(
        self,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ) -> None:
        """
        Filter entries by duration.

        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        """
        original_count = len(self.entries)
        self.entries = [
            e for e in self.entries
            if min_duration <= (e.end_time - e.start_time) <= max_duration
        ]

        # Update indices
        for i, entry in enumerate(self.entries):
            entry.index = i + 1

        logger.info(f"Filtered entries by duration: {original_count} -> {len(self.entries)}")

    def export_srt(self, output_path: str | Path) -> None:
        """
        Export entries to SRT format.

        Args:
            output_path: Path to output SRT file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for entry in self.entries:
            lines.append(str(entry.index))
            lines.append(
                f"{self.format_srt_time(entry.start_time)} --> "
                f"{self.format_srt_time(entry.end_time)}"
            )

            # Add speaker prefix if available
            if entry.speaker:
                lines.append(f"{entry.speaker}: {entry.text}")
            else:
                lines.append(entry.text)

            lines.append("")  # Blank line between entries

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Exported {len(self.entries)} entries to {output_path}")

    def get_batch_segments(
        self,
        group_by_speaker: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get subtitle segments formatted for batch TTS processing.

        Args:
            group_by_speaker: Whether to group consecutive entries by speaker

        Returns:
            List of segment dictionaries for TTS processing
        """
        segments = []

        if group_by_speaker:
            # Group consecutive entries by speaker
            current_group = None
            current_entries = []

            for entry in self.entries:
                speaker = entry.speaker or "unknown"
                voice_id = entry.voice_id

                group_key = (speaker, voice_id)

                if current_group != group_key:
                    # Save previous group
                    if current_entries:
                        segments.append(self._create_segment(current_entries))

                    # Start new group
                    current_group = group_key
                    current_entries = [entry]
                else:
                    current_entries.append(entry)

            # Add final group
            if current_entries:
                segments.append(self._create_segment(current_entries))
        else:
            # Each entry is a separate segment
            for entry in self.entries:
                segments.append(self._create_segment([entry]))

        logger.info(f"Created {len(segments)} TTS segments")
        return segments

    def _create_segment(self, entries: List[SubtitleEntry]) -> Dict[str, Any]:
        """Create a TTS segment from a list of subtitle entries."""
        if not entries:
            return {}

        first = entries[0]
        last = entries[-1]

        # Combine text with pauses
        combined_text = " ... ".join(e.text for e in entries)

        return {
            "text": combined_text,
            "start_time": first.start_time,
            "end_time": last.end_time,
            "duration": last.end_time - first.start_time,
            "speaker": first.speaker,
            "voice_id": first.voice_id,
            "entry_indices": [e.index for e in entries],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the subtitle file."""
        if not self.entries:
            return {}

        durations = [e.end_time - e.start_time for e in self.entries]
        total_duration = self.entries[-1].end_time - self.entries[0].start_time if self.entries else 0

        # Count speakers
        speakers = {}
        for entry in self.entries:
            if entry.speaker:
                speakers[entry.speaker] = speakers.get(entry.speaker, 0) + 1

        return {
            "total_entries": len(self.entries),
            "total_duration": total_duration,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "num_speakers": len(speakers),
            "speakers": speakers,
        }


# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Process subtitle files for batch dubbing"
    )
    parser.add_argument("input", help="Input subtitle file (SRT or ASS)")
    parser.add_argument("output", help="Output subtitle file")
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="Minimum subtitle duration in seconds"
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0,
        help="Maximum subtitle duration in seconds"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge short entries with adjacent ones"
    )
    parser.add_argument(
        "--detect-speakers", action="store_true",
        help="Detect and label speakers"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show statistics"
    )

    args = parser.parse_args()

    processor = SubtitleProcessor()

    # Parse subtitle file
    if args.input.endswith(".srt"):
        processor.parse_srt(args.input)
    elif args.input.endswith((".ass", ".ssa")):
        processor.parse_ass(args.input)
    else:
        # Try SRT by default
        processor.parse_srt(args.input)

    # Detect speakers if requested
    if args.detect_speakers:
        processor.detect_characters()

    # Filter by duration
    processor.filter_by_duration(args.min_duration, args.max_duration)

    # Merge short entries if requested
    if args.merge:
        processor.merge_short_entries()

    # Optimize punctuation
    processor.optimize_punctuation()

    # Show statistics
    if args.stats:
        stats = processor.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    # Export
    processor.export_srt(args.output)
    print(f"Processed subtitle saved to: {args.output}")
