"""Start Cloudflare Quick Tunnel for Inter-City Two-Person Sign Language Chat.

Allows Person A and Person B in different cities to communicate over the
internet with zero configuration, zero ports to open, and zero account required.

Usage:
    python scripts/share_chat.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sign_language.chat.server import ChatServer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main() -> None:
    print("=" * 70)
    print("   TWO-PERSON SIGN LANGUAGE CHAT: INTER-CITY GLOBAL CONNECTION")
    print("=" * 70)

    cloudflared_path = shutil.which("cloudflared")
    if not cloudflared_path:
        # Check standard Windows path
        fallback = Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe")
        if fallback.exists():
            cloudflared_path = str(fallback)
        else:
            print("[ERROR] 'cloudflared' not found.")
            print("Please install cloudflared or place cloudflared.exe in PATH.")
            sys.exit(1)

    # 1. Start background ChatServer on port 8765
    print("\n[1/3] Starting local WebSocket server on port 8765...")
    server = ChatServer(host="0.0.0.0", port=8765)
    server.start_background()
    time.sleep(0.5)

    # 2. Launch cloudflared quick tunnel
    print("[2/3] Creating global tunnel via Cloudflare (Zero Login Required)...")
    cmd = [
        cloudflared_path,
        "tunnel",
        "--url",
        "http://127.0.0.1:8765",
    ]

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    tunnel_domain: str | None = None
    url_pattern = re.compile(r"https://([a-zA-Z0-9\-\.]+\.trycloudflare\.com)")

    # Read output until tunnel domain found
    start_time = time.time()
    while time.time() - start_time < 20:
        line = proc.stderr.readline()
        if not line:
            continue
        match = url_pattern.search(line)
        if match:
            tunnel_domain = match.group(1)
            break

    if not tunnel_domain:
        print("[ERROR] Could not extract tunnel domain within 20 seconds.")
        proc.terminate()
        sys.exit(1)

    # 3. Present instructions to user
    web_link = f"https://{tunnel_domain}"
    friend_cmd = f'python scripts/chat_client.py --host {tunnel_domain} --room SIGN123 --user "Person B"'

    print("\n" + "=" * 70)
    print("                  ★ GLOBAL TUNNEL IS ACTIVE ★")
    print("=" * 70)
    print(f"\n  Web Chat Link : {web_link}")
    print(f"  Room Code     : SIGN123")
    print("\n----------------------------------------------------------------------")
    print("SEND THIS LINK TO YOUR FRIEND (ANY BROWSER, NO PYTHON NEEDED):")
    print(f"\n  👉 {web_link} 👈\n")
    print("Your friend clicks the link in Chrome/Safari on their phone or Mac,")
    print("types their name, and joins the room instantly!")
    print("----------------------------------------------------------------------")
    print("On your PC:")
    print("  1. In 'app.py', enter Room: SIGN123, User: Person A")
    print("  2. Click 'Connect'")
    print("  3. Start camera & sign!")
    print("=" * 70)
    print("\nPress Ctrl+C anytime to close the tunnel.\n")

    # Try copying Web link to clipboard
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(web_link)
        r.update()
        r.destroy()
        print("✓ Web Chat Link has been copied to your clipboard!")
    except Exception:
        pass

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nClosing tunnel...")
        proc.terminate()
        print("Tunnel closed.")


if __name__ == "__main__":
    main()
