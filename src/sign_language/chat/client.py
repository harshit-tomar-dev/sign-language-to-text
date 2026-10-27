"""WebSocket Chat Client for real-time messaging between Person A and Person B."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable

import websockets

logger = logging.getLogger(__name__)


class ChatClient:
    """Async WebSocket chat client with thread-safe send and callbacks."""

    def __init__(
        self,
        on_message: Callable[[str, str, str, bool], None] | None = None,
        on_status: Callable[[bool, str, int], None] | None = None,
    ) -> None:
        self.on_message = on_message
        self.on_status = on_status

        self.server_url: str = "ws://localhost:8765"
        self.room: str = "SIGN123"
        self.sender: str = "Person A"

        self.is_connected: bool = False
        self._ws: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._outbox: asyncio.Queue | None = None

    def connect(
        self,
        server_url: str = "ws://localhost:8765",
        room: str = "SIGN123",
        sender: str = "Person A",
    ) -> None:
        """Connect to WebSocket server in background thread."""
        if self.is_connected:
            self.disconnect()

        self.server_url = server_url.strip()
        self.room = room.strip().upper() or "SIGN123"
        self.sender = sender.strip() or "Person"
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ChatClientThread")
        self._thread.start()

    def _notify_status(self, connected: bool, message: str, count: int = 0) -> None:
        self.is_connected = connected
        if self.on_status:
            try:
                self.on_status(connected, message, count)
            except Exception as e:
                logger.debug("Status callback exception: %s", e)

    def _notify_message(self, sender: str, text: str, timestamp: str, is_system: bool) -> None:
        if self.on_message:
            try:
                self.on_message(sender, text, timestamp, is_system)
            except Exception as e:
                logger.debug("Message callback exception: %s", e)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._client_task())
        except Exception as e:
            logger.debug("Client loop ended: %s", e)
        finally:
            self._notify_status(False, "Disconnected")

    async def _client_task(self) -> None:
        try:
            self._notify_status(False, "Connecting...")
            async with websockets.connect(self.server_url, close_timeout=1) as ws:
                self._ws = ws
                self._outbox = asyncio.Queue()
                self._notify_status(True, f"Connected (Room: {self.room})", 1)

                join_msg = json.dumps({
                    "type": "join",
                    "room": self.room,
                    "sender": self.sender,
                })
                await ws.send(join_msg)

                receiver = asyncio.create_task(self._receiver_task(ws))
                sender = asyncio.create_task(self._sender_task(ws))

                done, pending = await asyncio.wait(
                    [receiver, sender],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

        except (ConnectionRefusedError, OSError) as e:
            self._notify_status(False, f"Cannot reach server ({self.server_url})")
            logger.info("Connection failed: %s", e)
        except Exception as e:
            self._notify_status(False, "Disconnected")
            logger.info("Connection exception: %s", e)

    async def _receiver_task(self, ws: Any) -> None:
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    msg_type = data.get("type", "message")
                    sender = data.get("sender", "Anonymous")
                    text = data.get("text", "")
                    t_str = data.get("timestamp") or datetime.now().strftime("%H:%M:%S")
                    users_count = int(data.get("users_count", 1))

                    is_system = (msg_type == "system")
                    self._notify_status(True, f"Connected ({users_count} in room)", users_count)
                    self._notify_message(sender, text, t_str, is_system)
                except json.JSONDecodeError:
                    continue
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass

    async def _sender_task(self, ws: Any) -> None:
        try:
            while not self._stop_event.is_set():
                if self._outbox is not None:
                    msg_json = await self._outbox.get()
                    await ws.send(msg_json)
        except asyncio.CancelledError:
            pass

    def send_message(self, text: str) -> bool:
        """Enqueue message to be sent over WebSocket."""
        clean = text.strip()
        if not clean or not self.is_connected or self._loop is None or self._outbox is None:
            return False

        payload = json.dumps({
            "type": "message",
            "room": self.room,
            "sender": self.sender,
            "text": clean,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

        self._loop.call_soon_threadsafe(self._outbox.put_nowait, payload)
        return True

    def disconnect(self) -> None:
        """Gracefully disconnect from chat server."""
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            async def _close_ws():
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass

            try:
                fut = asyncio.run_coroutine_threadsafe(_close_ws(), self._loop)
                fut.result(timeout=0.5)
            except Exception:
                pass

            self._loop.call_soon_threadsafe(self._loop.stop)
        self.is_connected = False
        self._notify_status(False, "Disconnected")
