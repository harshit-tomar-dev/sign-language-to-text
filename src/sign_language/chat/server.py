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
from pathlib import Path
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
  <title>SignFlow — Sign Language Live Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/static/style.css">
  <style>
    :root {
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      --bg-sage: #E8F1EC;
      --bg-cream: #F6F8F3;
      --bg-lavender: #F1EEF5;
      --bg-gradient: linear-gradient(135deg, var(--bg-sage) 0%, var(--bg-cream) 45%, var(--bg-lavender) 100%);
      --teal-primary: #0F766E;
      --teal-hover: #0D5C56;
      --teal-subtle: rgba(15, 118, 110, 0.08);
      --teal-surface: #E6F3F1;
      --accent-yellow: #F6D860;
      --card-bg: rgba(255, 255, 255, 0.68);
      --card-bg-solid: #FFFFFF;
      --card-border: rgba(255, 255, 255, 0.85);
      --card-border-subtle: rgba(20, 35, 31, 0.07);
      --card-shadow: 0 16px 40px rgba(20, 35, 31, 0.05), 0 2px 6px rgba(20, 35, 31, 0.02);
      --radius-card: 24px;
      --radius-inner: 16px;
      --radius-pill: 9999px;
      --text-heading: #14231F;
      --text-body: #2C3E38;
      --text-muted: #5A6E67;
      --text-subtle: #859891;
      --status-connected: #10B981;
      --status-disconnected: #94A3B8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-sans);
      background: var(--bg-gradient);
      background-attachment: fixed;
      color: var(--text-body);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }
    header {
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--card-border-subtle);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 38px;
      height: 38px;
      border-radius: var(--radius-inner);
      background: var(--teal-surface);
      color: var(--teal-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .brand-icon svg {
      width: 20px;
      height: 20px;
    }
    .header-title h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: var(--text-heading);
      line-height: 1.2;
    }
    .header-title span {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
      margin-top: 2px;
    }
    .main-wrapper {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 16px 24px 20px;
      gap: 14px;
      overflow: hidden;
    }
    .connection-drawer {
      background: rgba(255, 255, 255, 0.65);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-inner);
      box-shadow: 0 4px 16px rgba(20, 35, 31, 0.03);
      overflow: hidden;
    }
    .connection-drawer summary {
      list-style: none;
      cursor: pointer;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
    }
    .connection-drawer summary::-webkit-details-marker { display: none; }
    .connection-drawer summary:hover { color: var(--text-heading); }
    .conn-summary-left { display: flex; align-items: center; gap: 8px; }
    .conn-summary-left svg { width: 14px; height: 14px; }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--status-disconnected);
      transition: background 0.2s ease;
    }
    .status-badge.connected .status-dot { background: var(--status-connected); }
    .status-badge.connected { color: var(--teal-primary); font-weight: 600; }
    .conn-fields {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
      padding: 4px 18px 14px;
    }
    .input-pill {
      background: #FFFFFF;
      border: 1px solid var(--card-border-subtle);
      color: var(--text-heading);
      padding: 7px 14px;
      border-radius: var(--radius-pill);
      font-size: 12px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .input-pill:focus {
      border-color: var(--teal-primary);
      box-shadow: 0 0 0 3px var(--teal-subtle);
    }
    .btn-pill {
      background: var(--teal-primary);
      color: #FFFFFF;
      border: none;
      padding: 7px 18px;
      border-radius: var(--radius-pill);
      font-size: 12px;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease, transform 0.15s ease;
    }
    .btn-pill:hover {
      background: var(--teal-hover);
      transform: translateY(-1px);
    }
    #video-grid {
      display: flex;
      gap: 16px;
      justify-content: center;
      flex-wrap: wrap;
    }
    .video-card {
      position: relative;
      width: 340px;
      height: 195px;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-card);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: var(--card-shadow);
    }
    .video-card video, .video-card img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: var(--radius-card);
    }
    .video-placeholder {
      font-size: 12px;
      color: var(--text-muted);
      text-align: center;
      padding: 16px;
    }
    .video-chip {
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      padding: 4px 12px;
      border-radius: var(--radius-pill);
      font-size: 11px;
      font-weight: 600;
      color: var(--text-heading);
      border: 1px solid var(--card-border);
      box-shadow: 0 2px 8px rgba(20, 35, 31, 0.05);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .video-chip-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--teal-primary);
    }
    #chat-card {
      flex: 1;
      background: #111822;
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid #1E293B;
      border-radius: var(--radius-card);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #chat-container {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: #111822;
    }
    .msg-row {
      display: flex;
      flex-direction: column;
      max-width: 72%;
      animation: fadeIn 0.18s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .msg-row.incoming { align-self: flex-start; }
    .msg-row.outgoing { align-self: flex-end; align-items: flex-end; }
    .msg-meta {
      font-size: 11px;
      color: #94A3B8;
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .msg-bubble {
      padding: 10px 16px;
      font-size: 14px;
      line-height: 1.45;
      word-break: break-word;
    }
    .msg-row.incoming .msg-bubble {
      background: #182230;
      color: #F8FAFC;
      border: 1px solid #283548;
      border-radius: 18px 18px 18px 4px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
    }
    .msg-row.outgoing .msg-bubble {
      background: var(--teal-primary);
      color: #FFFFFF;
      border-radius: 18px 18px 4px 18px;
      box-shadow: 0 6px 18px rgba(15, 118, 110, 0.35);
    }
    .sign-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      font-size: 10px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: var(--radius-pill);
      background: rgba(15, 118, 110, 0.25);
      color: #5EEAD4;
      margin-right: 6px;
    }
    .msg-system {
      align-self: center;
      background: #1E293B;
      border: 1px solid #334155;
      color: #94A3B8;
      padding: 4px 16px;
      border-radius: var(--radius-pill);
      font-size: 11px;
      font-weight: 500;
    }
    footer {
      background: #0E141D;
      border-top: 1px solid #1E293B;
      padding: 14px 20px;
      display: flex;
      gap: 12px;
      align-items: center;
    }
    footer input {
      flex: 1;
      background: #182230;
      border: 1px solid #283548;
      color: #F8FAFC;
      padding: 12px 20px;
      border-radius: var(--radius-pill);
      font-size: 14px;
      font-family: inherit;
      outline: none;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    footer input::placeholder {
      color: #64748B;
    }
    footer input:focus {
      border-color: #14B8A6;
      box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.2);
    }
    footer button {
      background: var(--teal-primary);
      color: white;
      border: none;
      padding: 12px 26px;
      border-radius: var(--radius-pill);
      font-size: 13px;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.15s ease, transform 0.15s ease;
    }
    footer button:hover {
      background: var(--teal-hover);
      transform: translateY(-1px);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand-wrap">
      <div class="brand-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"/>
          <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/>
          <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v8"/>
          <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
        </svg>
      </div>
      <div class="header-title">
        <h1>SignFlow</h1>
        <span>SRM Institute of Science and Technology · Harshit Tomar (RA2411003030026) · Bhavishya · Shivank</span>
      </div>
    </div>
  </header>

  <div class="main-wrapper">
    <details class="connection-drawer">
      <summary>
        <div class="conn-summary-left">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
          <span>Connection Settings</span>
        </div>
        <div id="statusBadge" class="status-badge disconnected">
          <span class="status-dot"></span>
          <span id="statusText">Disconnected</span>
        </div>
      </summary>
      <div class="conn-fields">
        <input id="userName" class="input-pill" type="text" value="Person B" placeholder="Your Name" style="width: 120px;">
        <input id="roomCode" class="input-pill" type="text" value="SIGN123" placeholder="Room" style="width: 95px;">
        <button id="btnConnect" class="btn-pill" onclick="toggleConnection()">Reconnect</button>
      </div>
    </details>

    <div id="video-grid">
      <div class="video-card">
        <img id="remoteVideo" style="width: 100%; height: 100%; object-fit: cover; display: none;" />
        <span id="remotePlaceholder" class="video-placeholder">Waiting for remote camera stream...</span>
        <div class="video-chip"><span class="video-chip-dot"></span> Person A</div>
      </div>
      <div class="video-card">
        <video id="localVideo" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover;"></video>
        <div class="video-chip"><span class="video-chip-dot"></span> Your Camera</div>
      </div>
    </div>

    <div id="chat-card">
      <div id="chat-container"></div>
      <footer>
        <input id="inputMessage" type="text" placeholder="Type a message..." onkeydown="handleKeyDown(event)">
        <button onclick="sendMessage()">Send</button>
      </footer>
    </div>
  </div>

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

    function updateStatus(connected, text) {
      const badge = document.getElementById("statusBadge");
      const statusText = document.getElementById("statusText");
      if (connected) {
        badge.className = "status-badge connected";
        if (statusText) statusText.textContent = text || "Connected";
      } else {
        badge.className = "status-badge disconnected";
        if (statusText) statusText.textContent = text || "Disconnected";
      }
    }

    function initConnection() {
      const url = getWsUrl();
      updateStatus(false, "Connecting...");

      try {
        ws = new WebSocket(url);
      } catch (e) {
        updateStatus(false, "Disconnected");
        return;
      }

      ws.onopen = () => {
        isConnected = true;
        updateStatus(true, "Connected");

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
        updateStatus(false, "Disconnected");
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };
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
        bubble.innerHTML = `<span class="sign-badge">Sign</span> ${data.text}`;
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

    function handleKeyDown(event) {
      if (event.key === "Enter") {
        sendMessage();
      }
    }

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

        # Serve static CSS stylesheet
        if hasattr(request, "path") and request.path in ("/static/style.css", "/style.css"):
            css_path = Path(__file__).parent / "static" / "style.css"
            if css_path.exists():
                css_content = css_path.read_text(encoding="utf-8")
                resp = connection.respond(http.HTTPStatus.OK, css_content)
                resp.headers["Content-Type"] = "text/css; charset=utf-8"
                return resp

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
