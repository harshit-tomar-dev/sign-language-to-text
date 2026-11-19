"""Interactive CLI Chat Client for Two-Person Sign Language Conversation.

Allows Person B (friend on Mac, Linux, or Windows) to connect to Person A's
room in real time without needing a camera or GUI.

Usage:
    # Connect locally:
    python scripts/chat_client.py --user "Person B"

    # Connect to friend's PC on Wi-Fi:
    python scripts/chat_client.py --host 192.168.1.39 --room SIGN123 --user "Person B"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.chat.client import ChatClient

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-Person Sign Language Chat CLI Client"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host IP address of the server (e.g. 192.168.1.39 or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--room",
        default="SIGN123",
        help="Room code to join (default: SIGN123)",
    )
    parser.add_argument(
        "--user",
        default="Person B",
        help="Display name for this user (default: Person B)",
    )
    args = parser.parse_args()

    raw_host = args.host.strip()
    cleaned_host = (
        raw_host.replace("https://", "")
        .replace("http://", "")
        .replace("wss://", "")
        .replace("ws://", "")
        .rstrip("/")
    )
    port = args.port
    room = args.room.strip().upper()
    user = args.user.strip()

    if raw_host.startswith("wss://") or raw_host.startswith("ws://"):
        server_url = raw_host
    elif (
        ".trycloudflare.com" in cleaned_host
        or ".ngrok" in cleaned_host
        or ".loca.lt" in cleaned_host
    ):
        server_url = f"wss://{cleaned_host}"
    else:
        server_url = f"ws://{cleaned_host}:{port}"

    print("=" * 65)
    print("     TWO-PERSON SIGN LANGUAGE LIVE CHAT CLIENT")
    print(f"     Connecting to : {server_url}")
    print(f"     Room Code     : {room}")
    print(f"     Username      : {user}")
    print("=" * 65)
    print("Type your message and press ENTER to send.")
    print("Type 'exit' or press Ctrl+C to leave the room.\n")

    def on_message(sender: str, text: str, timestamp: str, is_system: bool) -> None:
        if is_system:
            print(f"\n[SYSTEM {timestamp}] {text}")
        else:
            print(f"\n[{timestamp}] {sender}: {text}")
        print(f"{user} > ", end="", flush=True)

    def on_status(connected: bool, status_msg: str, count: int) -> None:
        if connected:
            print(f"✓ Connected to room '{room}'! (Active users in room: {count})")
            print(f"{user} > ", end="", flush=True)
        else:
            print(f"\n[INFO] {status_msg}")

    client = ChatClient(on_message=on_message, on_status=on_status)
    client.connect(server_url=server_url, room=room, sender=user)

    # Wait briefly for connection
    time.sleep(0.5)

    try:
        while True:
            try:
                msg = input().strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not msg:
                continue
            if msg.lower() in ("exit", "quit"):
                break

            if not client.is_connected:
                print("[WARNING] Not connected to server yet. Trying to send...")
            client.send_message(msg)

    finally:
        print("\nLeaving room...")
        client.disconnect()
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    main()
