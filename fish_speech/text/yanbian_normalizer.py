"""
Yanbian Korean text normalizer for TTS preprocessing.

This module handles text normalization specific to Yanbian Korean dialect,
which is spoken by the Korean minority in Yanbian, China.

Features:
- Yanbian-specific vocabulary mapping
- Chinese-Korean mixed text handling
- Number and date formatting
- Punctuation normalization
"""

import re
from typing import Dict, List, Optional

from loguru import logger


class YanbianKoreanNormalizer:
    """
    Text normalizer for Yanbian Korean dialect.

    Yanbian Korean differs from standard Korean (South Korea) in:
    - Vocabulary influenced by Chinese
    - Pronunciation patterns
    - Mixed Chinese character usage
    - Regional expressions
    """

    def __init__(self):
        """Initialize the normalizer with Yanbian-specific mappings."""

        # Yanbian-specific vocabulary mappings
        # (Standard Korean -> Yanbian Korean)
        self.yanbian_vocab: Dict[str, str] = {
            # Common words with Yanbian variants
            "안녕하세요": "안녕하십니까",
            "감사합니다": "고맙습니다",
            "미안합니다": "죄송합니다",
            "그래요": "그렇습니다",
            "아니요": "아닙니다",

            # Chinese-influenced terms
            "학교": "쥐교",  # 学校
            "회사": "꾸스",  # 公司
            "은행": "인환",  # 银行
            "병원": "뵤원",  # 医院

            # Regional expressions
            "정말": "참",
            "아주": "되게",
            "조금": "좀",
            "많이": "많이",
            "조용히": "즈시히",
        }

        # Number to Korean text mappings (Yanbian pronunciation)
        self.number_mappings: Dict[str, str] = {
            "0": "공",
            "1": "일",
            "2": "이",
            "3": "삼",
            "4": "사",
            "5": "오",
            "6": "육",
            "7": "칠",
            "8": "팔",
            "9": "구",
            "10": "십",
        }

        # Chinese punctuation to Korean conversions
        self.punctuation_map: Dict[str, str] = {
            "。": ".",
            "，": ",",
            "！": "!",
            "？": "?",
            "：": ":",
            "；": ";",
            "（": "(",
            "）": ")",
            "「": "'",
            "」": "'",
            "『": '"',
            "』": '"',
        }

        # Common Chinese phrases that might appear in Yanbian Korean context
        self.chinese_phrases: Dict[str, str] = {
            "你好": "안녕하세요",
            "谢谢": "감사합니다",
            "对不起": "죄송합니다",
            "再见": "안녕히 가세요",
            "是": "예",
            "不是": "아니요",
        }

    def normalize_text(
        self,
        text: str,
        apply_vocab_mapping: bool = True,
        normalize_numbers: bool = True,
        normalize_punctuation: bool = True,
    ) -> str:
        """
        Normalize text for Yanbian Korean TTS.

        Args:
            text: Input text
            apply_vocab_mapping: Whether to apply vocabulary mappings
            normalize_numbers: Whether to normalize numbers
            normalize_punctuation: Whether to normalize punctuation

        Returns:
            Normalized text
        """
        result = text

        # Normalize whitespace
        result = " ".join(result.split())

        # Apply punctuation normalization
        if normalize_punctuation:
            result = self.normalize_punctuation(result)

        # Apply vocabulary mappings
        if apply_vocab_mapping:
            result = self.apply_vocab_mapping(result)

        # Normalize numbers
        if normalize_numbers:
            result = self.normalize_numbers(result)

        # Handle Chinese-Korean mixed text
        result = self.handle_mixed_text(result)

        # Final cleanup
        result = self.cleanup_text(result)

        return result

    def normalize_punctuation(self, text: str) -> str:
        """Convert Chinese punctuation to Korean/Western equivalents."""
        result = text
        for chinese, korean in self.punctuation_map.items():
            result = result.replace(chinese, korean)
        return result

    def apply_vocab_mapping(self, text: str) -> str:
        """Apply Yanbian-specific vocabulary mappings."""
        result = text

        # Apply longer phrases first to avoid partial replacements
        sorted_vocab = sorted(
            self.yanbian_vocab.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for standard, yanbian in sorted_vocab:
            result = re.sub(
                r"\b" + re.escape(standard) + r"\b",
                yanbian,
                result,
            )

        return result

    def normalize_numbers(self, text: str) -> str:
        """
        Convert Arabic numerals to Korean text.

        This handles basic numbers. For production, consider using
        dedicated number normalization libraries.
        """
        result = text

        # Simple number conversion (for standalone numbers)
        def replace_number(match):
            num = match.group(0)
            return self.number_mappings.get(num, num)

        result = re.sub(r"\d+", replace_number, result)

        return result

    def handle_mixed_text(self, text: str) -> str:
        """
        Handle Chinese characters mixed with Korean text.

        In Yanbian Korean context, Chinese characters may appear.
        This attempts to convert common phrases to Korean.
        """
        result = text

        # Apply Chinese phrase conversions
        for chinese, korean in self.chinese_phrases.items():
            result = result.replace(chinese, korean)

        # Handle individual Chinese characters
        # (For production, use a proper Chinese-Korean dictionary)
        result = self.convert_chinese_characters(result)

        return result

    def convert_chinese_characters(self, text: str) -> str:
        """
        Convert Chinese characters to Korean Hangul.

        This is a placeholder. For production, use a proper
        Chinese-Hanja-Korean conversion library.

        Current approach: Keep Chinese characters as-is since
        Fish Speech can handle them, but could add conversion
        in the future.
        """
        # Check if text contains Chinese characters
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))

        if has_chinese:
            logger.debug("Text contains Chinese characters, keeping as-is")

        return text

    def cleanup_text(self, text: str) -> str:
        """Final cleanup of normalized text."""
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def add_emotion_markers(
        self,
        text: str,
        emotion: Optional[str] = None,
    ) -> str:
        """
        Add emotion markers to text for Fish Speech TTS.

        Fish Speech supports emotion markers like (angry), (sad), (happy).

        Args:
            text: Input text
            emotion: Emotion to apply (angry, sad, happy, excited, etc.)

        Returns:
            Text with emotion markers
        """
        if emotion is None:
            return text

        # Map emotion names to Fish Speech markers
        emotion_map = {
            "angry": "(angry)",
            "sad": "(sad)",
            "happy": "(happy)",
            "excited": "(excited)",
            "surprised": "(surprised)",
            "whisper": "(whispering)",
            "shouting": "(shouting)",
            "nervous": "(nervous)",
            "confident": "(confident)",
        }

        marker = emotion_map.get(emotion.lower())

        if marker:
            # Add marker at the beginning of text
            return f"{marker} {text}"

        return text

    def split_by_sentences(
        self,
        text: str,
        max_length: int = 200,
    ) -> List[str]:
        """
        Split text into sentences for TTS processing.

        Args:
            text: Input text
            max_length: Maximum characters per sentence

        Returns:
            List of sentences
        """
        # Korean sentence-ending patterns
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n",
                           ".!", "!.", "?!", "♥", "♪"]

        # Split by sentence endings
        sentences = []
        current = ""

        for char in text:
            current += char

            # Check if we hit a sentence ending
            if any(current.endswith(ending) for ending in sentence_endings):
                sentences.append(current.strip())
                current = ""

            # Check max length
            elif len(current) >= max_length:
                # Force split at space if possible
                if " " in current:
                    split_point = current.rindex(" ")
                    sentences.append(current[:split_point].strip())
                    current = current[split_point + 1:]
                else:
                    sentences.append(current.strip())
                    current = ""

        # Add remaining text
        if current.strip():
            sentences.append(current.strip())

        return [s for s in sentences if s]

    def process_for_tts(
        self,
        text: str,
        emotion: Optional[str] = None,
        max_sentence_length: int = 200,
    ) -> List[str]:
        """
        Complete preprocessing pipeline for TTS.

        Args:
            text: Input text
            emotion: Optional emotion to apply
            max_sentence_length: Maximum characters per sentence

        Returns:
            List of processed sentences ready for TTS
        """
        # Normalize text
        normalized = self.normalize_text(text)

        # Add emotion markers
        if emotion:
            normalized = self.add_emotion_markers(normalized, emotion)

        # Split into sentences
        sentences = self.split_by_sentences(normalized, max_sentence_length)

        logger.debug(f"Processed text into {len(sentences)} sentences")

        return sentences


# Convenience function
def normalize_yanbian_korean(
    text: str,
    emotion: Optional[str] = None,
) -> str:
    """
    Convenience function to normalize Yanbian Korean text.

    Args:
        text: Input text
        emotion: Optional emotion marker

    Returns:
        Normalized text
    """
    normalizer = YanbianKoreanNormalizer()

    if emotion:
        text = normalizer.add_emotion_markers(text, emotion)

    return normalizer.normalize_text(text)


# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Normalize Yanbian Korean text for TTS"
    )
    parser.add_argument("input", help="Input text or file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument(
        "--emotion", choices=["angry", "sad", "happy", "excited", "surprised"],
        help="Emotion to apply"
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format"
    )

    args = parser.parse_args()

    normalizer = YanbianKoreanNormalizer()

    # Read input
    input_path = Path(args.input)
    if input_path.exists():
        text = input_path.read_text(encoding="utf-8")
    else:
        text = args.input

    # Process
    sentences = normalizer.process_for_tts(text, emotion=args.emotion)

    # Output
    if args.format == "json":
        output = json.dumps(sentences, ensure_ascii=False, indent=2)
    else:
        output = "\n".join(sentences)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output saved to: {args.output}")
    else:
        print(output)
