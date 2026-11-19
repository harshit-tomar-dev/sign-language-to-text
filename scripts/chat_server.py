"""Standalone Real-Time WebSocket Chat Server for Sign Language System.

Usage:
    python scripts/chat_server.py [--host 0.0.0.0] [--port 8765]

Allows Person A and Person B on same computer or local network to exchange
real-time converted sign language text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.chat.server import start_server_standalone


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign Language Chat WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    args = parser.parse_args()

    start_server_standalone(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
