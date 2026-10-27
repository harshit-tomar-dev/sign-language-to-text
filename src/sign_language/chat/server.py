"""WebSocket Real-Time Chat Server with Room Support.

Allows multiple users (e.g. Person A and Person B) to join a specific
Room ID (e.g. SIGN123) and exchange text and sign conversion messages in real time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any

import websockets

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765


class ChatServer:
    """Manages WebSocket connections and message routing across isolated rooms."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        # Map: room_id -> dict of {websocket: sender_name}
        self.rooms: dict[str, dict[Any, str]] = {}
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.is_running = False

    async def _handler(self, websocket: Any) -> None:
        current_room: str | None = None
        current_sender = "Anonymous"

        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "message")
                room = data.get("room", "DEFAULT").strip().upper()
                sender = data.get("sender", "Anonymous").strip()

                if msg_type == "join":
                    current_room = room
                    current_sender = sender
                    if current_room not in self.rooms:
                        self.rooms[current_room] = {}
                    self.rooms[current_room][websocket] = current_sender

                    # Notify room members
                    count = len(self.rooms[current_room])
                    welcome_payload = json.dumps({
                        "type": "system",
                        "room": current_room,
                        "sender": "SYSTEM",
                        "text": f"{current_sender} joined room {current_room}.",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "users_count": count,
                    })
                    await self._broadcast_room(current_room, welcome_payload)

                elif msg_type == "message":
                    text = data.get("text", "").strip()
                    if not text:
                        continue

                    timestamp = data.get("timestamp") or datetime.now().strftime("%H:%M:%S")
                    count = len(self.rooms.get(room, {}))
                    payload = json.dumps({
                        "type": "message",
                        "room": room,
                        "sender": sender,
                        "text": text,
                        "timestamp": timestamp,
                        "users_count": count,
                    })
                    await self._broadcast_room(room, payload)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if current_room and current_room in self.rooms:
                if websocket in self.rooms[current_room]:
                    del self.rooms[current_room][websocket]
                    count = len(self.rooms[current_room])
                    leave_payload = json.dumps({
                        "type": "system",
                        "room": current_room,
                        "sender": "SYSTEM",
                        "text": f"{current_sender} left room {current_room}.",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "users_count": count,
                    })
                    await self._broadcast_room(current_room, leave_payload)

                if not self.rooms[current_room]:
                    del self.rooms[current_room]

    async def _broadcast_room(self, room: str, message: str) -> None:
        if room not in self.rooms:
            return

        dead_sockets = []
        for ws in list(self.rooms[room].keys()):
            try:
                await ws.send(message)
            except Exception:
                dead_sockets.append(ws)

        for ws in dead_sockets:
            if ws in self.rooms[room]:
                del self.rooms[room][ws]

    async def _serve(self) -> None:
        async with websockets.serve(self._handler, self.host, self.port) as server:
            self._server = server
            self.is_running = True
            logger.info("ChatServer running on ws://%s:%d", self.host, self.port)
            await asyncio.Future()  # Run forever

    def start_background(self) -> None:
        """Start the WebSocket server in a separate background daemon thread."""
        if self.is_running:
            return

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._serve())
            except Exception as e:
                logger.debug("Server loop ended: %s", e)
            finally:
                self.is_running = False

        self._thread = threading.Thread(target=run, daemon=True, name="ChatServerThread")
        self._thread.start()


def start_server_standalone(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run server synchronously in current process."""
    server = ChatServer(host, port)
    print(f"Starting Sign Language Chat Server on ws://{host}:{port}...")
    print("Press Ctrl+C to terminate server.\n")
    asyncio.run(server._serve())
