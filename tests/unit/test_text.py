"""Unit tests for Sign to Text Conversion."""

import pytest
from sign_language.text.composer import TextBuffer, sign_to_word


def test_sign_to_word_mapping():
    assert sign_to_word("HELLO") == "Hello"
    assert sign_to_word("THANK_YOU") == "Thank you"
    assert sign_to_word("PLEASE") == "Please"
    assert sign_to_word("YES") == "Yes"
    assert sign_to_word("NO") == "No"
    assert sign_to_word("SORRY") == "Sorry"
    assert sign_to_word("HELP") == "Help"
    assert sign_to_word("I_LOVE_YOU") == "I love you"
    assert sign_to_word("_NEUTRAL") == ""
    assert sign_to_word("UNCERTAIN") == ""
    assert sign_to_word("Not trained") == ""


def test_text_buffer_operations():
    buf = TextBuffer()
    assert buf.text == ""
    assert len(buf) == 0

    # Add words
    buf.add_sign("HELLO")
    assert buf.text == "Hello"
    assert len(buf) == 1

    # Neutral doesn't emit
    buf.add_sign("_NEUTRAL")
    assert buf.text == "Hello"

    buf.add_word("my")
    buf.add_word("name")
    buf.add_word("is")
    buf.add_word("Harshit")

    assert buf.text == "Hello my name is Harshit"
    assert len(buf) == 5

    # Backspace
    removed = buf.backspace()
    assert removed == "Harshit"
    assert buf.text == "Hello my name is"

    # Clear
    buf.clear()
    assert buf.text == ""
    assert len(buf) == 0
