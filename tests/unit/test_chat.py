"""Unit tests for WebSocket Real-Time Chat Server and Client."""

import time
import pytest
from sign_language.chat.server import ChatServer
from sign_language.chat.client import ChatClient


def test_chat_server_and_client_roundtrip():
    # Start local test server on alternate port
    server = ChatServer(host="127.0.0.1", port=8799)
    server.start_background()
    time.sleep(0.15)

    received_messages_a = []
    received_messages_b = []

    client_a = ChatClient(
        on_message=lambda sender, text, ts, is_sys: received_messages_a.append((sender, text))
    )
    client_b = ChatClient(
        on_message=lambda sender, text, ts, is_sys: received_messages_b.append((sender, text))
    )

    try:
        # Both connect to same Room SIGN_TEST
        client_a.connect("ws://127.0.0.1:8799", room="SIGN_TEST", sender="Person A")
        client_b.connect("ws://127.0.0.1:8799", room="SIGN_TEST", sender="Person B")
        time.sleep(0.3)

        assert client_a.is_connected
        assert client_b.is_connected

        # Person A sends a message
        client_a.send_message("Hello from Person A")
        time.sleep(0.2)

        # Check Person B received it
        texts_b = [msg[1] for msg in received_messages_b]
        assert "Hello from Person A" in texts_b

        # Person B sends a reply
        client_b.send_message("Nice to meet you!")
        time.sleep(0.2)

        # Check Person A received it
        texts_a = [msg[1] for msg in received_messages_a]
        assert "Nice to meet you!" in texts_a

    finally:
        client_a.disconnect()
        client_b.disconnect()
