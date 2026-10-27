"""Sign to Text Composer for converting discrete sign classifications into readable sentences."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Canonical human-readable mappings for vocabulary
SIGN_TO_WORD_MAP: dict[str, str] = {
    "HELLO": "Hello",
    "THANK_YOU": "Thank you",
    "PLEASE": "Please",
    "YES": "Yes",
    "NO": "No",
    "SORRY": "Sorry",
    "HELP": "Help",
    "NAME": "Name",
    "WATER": "Water",
    "EAT": "Eat",
    "I_LOVE_YOU": "I love you",
    "YOU": "You",
    "_NEUTRAL": "",  # Neutral does not emit text
}


def sign_to_word(sign_label: str) -> str:
    """Convert raw sign class label into formatted readable English text.

    Args:
        sign_label: Class label from model (e.g. 'THANK_YOU', 'HELLO').

    Returns:
        Formatted text (e.g. 'Thank you', 'Hello'). Returns empty string for '_NEUTRAL'.
    """
    cleaned = sign_label.strip()
    if cleaned in SIGN_TO_WORD_MAP:
        return SIGN_TO_WORD_MAP[cleaned]

    if cleaned.startswith("_") or cleaned == "UNCERTAIN" or cleaned == "Not trained":
        return ""

    # Fallback: title-case with underscores replaced by spaces
    return cleaned.replace("_", " ").capitalize()


class TextBuffer:
    """Maintains and manages a sequence of recognized sign words into a coherent text."""

    def __init__(self) -> None:
        self._words: list[str] = []

    def add_sign(self, sign_label: str) -> str | None:
        """Convert sign and append word to buffer if valid.

        Returns:
            The added word if appended, or None if empty/neutral.
        """
        word = sign_to_word(sign_label)
        if not word:
            return None

        self._words.append(word)
        return word

    def add_word(self, word: str) -> None:
        """Add arbitrary text word to buffer."""
        cleaned = word.strip()
        if cleaned:
            self._words.append(cleaned)

    def backspace(self) -> str | None:
        """Remove the most recently added word."""
        if self._words:
            return self._words.pop()
        return None

    def clear(self) -> None:
        """Clear all composed words."""
        self._words.clear()

    @property
    def words(self) -> list[str]:
        """Return list of current words."""
        return list(self._words)

    @property
    def text(self) -> str:
        """Format the current buffer as a clean readable sentence string."""
        if not self._words:
            return ""

        # Join with space
        joined = " ".join(self._words)
        return joined[0].upper() + joined[1:] if len(joined) > 1 else joined.upper()

    def __len__(self) -> int:
        return len(self._words)

    def __str__(self) -> str:
        return self.text
