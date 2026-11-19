"""WebSocket Real-Time Chat Server with Room Support.

Allows multiple users (e.g. Person A and Person B) to join a specific
Room ID (e.g. SIGN123) and exchange text and sign conversion messages in real time.
"""

from __future__ import annotations

import asyncio
import http
import json
import logging
import threading
from datetime import datetime
from typing import Any

import websockets

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

WEB_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign Language Live Chat — SRMIST</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0F172A;
      color: #F8FAFC;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #1E293B;
      border-bottom: 1px solid #334155;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .header-title {
      display: flex;
      flex-direction: column;
    }
    .header-title h1 {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: #F8FAFC;
    }
    .header-title span {
      font-size: 11px;
      color: #94A3B8;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: #10B981;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.disconnected {
      background: rgba(239, 68, 68, 0.15);
      color: #EF4444;
      border-color: rgba(239, 68, 68, 0.3);
    }
    .top-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .top-controls input {
      background: #020617;
      border: 1px solid #334155;
      color: #F8FAFC;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
    }
    .top-controls button {
      background: #2563EB;
      color: white;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    .top-controls button:hover { background: #1D4ED8; }
    #chat-container {
      flex: 1;
      padding: 18px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg-row {
      display: flex;
      flex-direction: column;
      max-width: 80%;
      animation: fadeIn 0.15s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .msg-row.incoming { align-self: flex-start; }
    .msg-row.outgoing { align-self: flex-end; align-items: flex-end; }
    .msg-meta {
      font-size: 10px;
      color: #94A3B8;
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .msg-bubble {
      padding: 10px 14px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.4;
      word-break: break-word;
    }
    .msg-row.incoming .msg-bubble {
      background: #1E293B;
      color: #F8FAFC;
      border: 1px solid #334155;
      border-top-left-radius: 2px;
    }
    .msg-row.outgoing .msg-bubble {
      background: #2563EB;
      color: #FFFFFF;
      border-top-right-radius: 2px;
    }
    .msg-bubble.sign-tag {
      border-left: 3px solid #10B981;
    }
    .sign-badge {
      display: inline-block;
      font-size: 9px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 1px 5px;
      border-radius: 4px;
      background: rgba(16, 185, 129, 0.2);
      color: #34D399;
      margin-right: 4px;
    }
    .msg-system {
      align-self: center;
      background: rgba(51, 65, 85, 0.5);
      color: #94A3B8;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 11px;
    }
    footer {
      background: #1E293B;
      border-top: 1px solid #334155;
      padding: 12px 18px;
      display: flex;
      gap: 10px;
    }
    footer input {
      flex: 1;
      background: #020617;
      border: 1px solid #334155;
      color: #F8FAFC;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }
    footer input:focus { border-color: #2563EB; }
    footer button {
      background: #2563EB;
      color: white;
      border: none;
      padding: 0 20px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    footer button:hover { background: #1D4ED8; }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h1>Sign Language Live Chat & Video Call</h1>
      <span>SRM Institute of Science and Technology · Academic Live Demo</span>
    </div>
    <div class="top-controls">
      <input id="userName" type="text" value="Person B" placeholder="Your Name" style="width: 100px;">
      <input id="roomCode" type="text" value="SIGN123" placeholder="Room" style="width: 80px;">
      <button id="btnConnect" onclick="toggleConnection()">Reconnect</button>
      <div id="statusBadge" class="status-badge">Connecting...</div>
    </div>
  </header>

  <!-- Dual Camera Video Bar: Harshit's Live Camera + Friend's Live Camera -->
  <div id="video-grid" style="display: flex; gap: 12px; padding: 10px 18px; background: #020617; border-bottom: 1px solid #334155; justify-content: center; flex-wrap: wrap;">
    <div style="position: relative; width: 320px; height: 190px; background: #0F172A; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; align-items: center; justify-content: center;">
      <img id="remoteVideo" style="width: 100%; height: 100%; object-fit: cover; display: none;" />
      <span id="remotePlaceholder" style="font-size: 12px; color: #94A3B8; text-align: center; padding: 10px;">Waiting for Harshit's Live Camera...</span>
      <div style="position: absolute; bottom: 8px; left: 8px; background: rgba(15, 23, 42, 0.85); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: #F8FAFC;">Harshit (Person A)</div>
    </div>
    <div style="position: relative; width: 320px; height: 190px; background: #0F172A; border-radius: 8px; overflow: hidden; border: 1px solid #334155; display: flex; align-items: center; justify-content: center;">
      <video id="localVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover;"></video>
      <div style="position: absolute; bottom: 8px; left: 8px; background: rgba(15, 23, 42, 0.85); padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; color: #F8FAFC;">Your Camera (Person B)</div>
    </div>
  </div>

  <div id="chat-container"></div>

  <footer>
    <input id="inputMessage" type="text" placeholder="Type a message to Harshit..." onkeydown="handleKeyDown(event)">
    <button onclick="sendMessage()">Send</button>
  </footer>

  <script>
    let ws = null;
    let isConnected = false;
    let localStream = null;
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");

    function getWsUrl() {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${window.location.host}`;
    }

    async function startCamera() {
      try {
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 320 }, height: { ideal: 240 } },
          audio: false
        });
        const localVideo = document.getElementById("localVideo");
        localVideo.srcObject = localStream;

        // Send video frame every 120ms
        setInterval(() => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          if (localVideo.videoWidth > 0 && localVideo.videoHeight > 0) {
            ctx.drawImage(localVideo, 0, 0, 320, 240);
            const dataUrl = canvas.toDataURL("image/jpeg", 0.5);
            const base64 = dataUrl.split(",")[1];
            const room = document.getElementById("roomCode").value.trim().toUpperCase() || "SIGN123";
            const sender = document.getElementById("userName").value.trim() || "Person B";
            ws.send(JSON.stringify({
              type: "video_frame",
              room: room,
              sender: sender,
              image: base64
            }));
          }
        }, 120);
      } catch (err) {
        console.log("Webcam not available:", err);
      }
    }

    function initConnection() {
      const url = getWsUrl();
      const statusBadge = document.getElementById("statusBadge");
      statusBadge.className = "status-badge disconnected";
      statusBadge.textContent = "Connecting...";

      try {
        ws = new WebSocket(url);
      } catch (e) {
        statusBadge.textContent = "Error";
        return;
      }

      ws.onopen = () => {
        isConnected = true;
        statusBadge.className = "status-badge";
        statusBadge.textContent = "● Connected";

        const room = document.getElementById("roomCode").value.trim().toUpperCase() || "SIGN123";
        const sender = document.getElementById("userName").value.trim() || "Person B";

        ws.send(JSON.stringify({
          type: "join",
          room: room,
          sender: sender
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "video_frame") {
            const remoteImg = document.getElementById("remoteVideo");
            const remotePlaceholder = document.getElementById("remotePlaceholder");
            remoteImg.src = "data:image/jpeg;base64," + data.image;
            remoteImg.style.display = "block";
            if (remotePlaceholder) remotePlaceholder.style.display = "none";
            return;
          }
          renderMessage(data);
        } catch (e) {
          console.error("Invalid message format:", event.data);
        }
      };

      ws.onclose = () => {
        isConnected = false;
        statusBadge.className = "status-badge disconnected";
        statusBadge.textContent = "○ Disconnected";
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };
    }
    }

    function toggleConnection() {
      if (ws) {
        ws.close();
      }
      setTimeout(initConnection, 200);
    }

    function renderMessage(data) {
      const container = document.getElementById("chat-container");
      const currentSender = document.getElementById("userName").value.trim();

      if (data.type === "system") {
        const sysDiv = document.createElement("div");
        sysDiv.className = "msg-system";
        sysDiv.textContent = `[${data.timestamp}] ${data.text}`;
        container.appendChild(sysDiv);
        container.scrollTop = container.scrollHeight;
        return;
      }

      const isMe = data.sender.toLowerCase() === currentSender.toLowerCase();
      const row = document.createElement("div");
      row.className = `msg-row ${isMe ? 'outgoing' : 'incoming'}`;

      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.innerHTML = `<strong>${data.sender}</strong> <span>${data.timestamp}</span>`;

      const bubble = document.createElement("div");
      bubble.className = "msg-bubble";

      if (data.sender.toLowerCase().includes("person a") || data.sender.toLowerCase().includes("harshit")) {
        bubble.innerHTML = `<span class="sign-badge">✋ Hand Sign</span> ${data.text}`;
      } else {
        bubble.textContent = data.text;
      }

      row.appendChild(meta);
      row.appendChild(bubble);
      container.appendChild(row);
      container.scrollTop = container.scrollHeight;
    }

    function sendMessage() {
      const input = document.getElementById("inputMessage");
      const text = input.value.trim();
      if (!text || !ws || !isConnected) return;

      const room = document.getElementById("roomCode").value.trim().toUpperCase() || "SIGN123";
      const sender = document.getElementById("userName").value.trim() || "Person B";

      ws.send(JSON.stringify({
        type: "message",
        room: room,
        sender: sender,
        text: text
      }));

      input.value = "";
      input.focus();
    }

    function handleKeyDown(e) {
      if (e.key === "Enter") {
        sendMessage();
      }
    }

    // Connect automatically and start camera when page loads
    window.addEventListener("DOMContentLoaded", () => {
      initConnection();
      startCamera();
    });
  </script>
</body>
</html>
"""


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

                elif msg_type == "video_frame":
                    # Video streaming frame: forward to all OTHER clients in the room
                    image_data = data.get("image", "")
                    if image_data and room in self.rooms:
                        frame_payload = json.dumps({
                            "type": "video_frame",
                            "room": room,
                            "sender": sender,
                            "image": image_data,
                        })
                        await self._broadcast_room(room, frame_payload, exclude=websocket)

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

    async def _broadcast_room(self, room: str, message: str, exclude: Any = None) -> None:
        if room not in self.rooms:
            return

        dead_sockets = []
        for ws in list(self.rooms[room].keys()):
            if exclude is not None and ws == exclude:
                continue
            try:
                await ws.send(message)
            except Exception:
                dead_sockets.append(ws)

        for ws in dead_sockets:
            if ws in self.rooms[room]:
                del self.rooms[room][ws]

    async def _process_request(self, connection: Any, request: Any) -> Any:
        # If this is a WebSocket handshake (Upgrade: websocket), proceed with WebSocket handler
        if hasattr(request, "headers"):
            upgrade = request.headers.get("Upgrade", "").lower()
            if upgrade == "websocket":
                return None

        # If client requested root or web chat in a browser via standard HTTP GET
        if hasattr(request, "path") and request.path in ("/", "/chat", "/index.html"):
            resp = connection.respond(http.HTTPStatus.OK, WEB_CHAT_HTML)
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        return None

    async def _serve(self) -> None:
        async with websockets.serve(
            self._handler,
            self.host,
            self.port,
            process_request=self._process_request,
        ) as server:
            self._server = server
            self.is_running = True
            logger.info("ChatServer running on ws://%s:%d (Web Chat: http://%s:%d)", self.host, self.port, self.host, self.port)
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
