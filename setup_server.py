#!/usr/bin/env python3
"""
ComfyUI Setup Server
====================
Lightweight web server that runs on RunPod to handle model/node installation.
Communicates with setupRunPodInstance.py running on your local machine.

Usage:
    python setup_server.py

Endpoints:
    GET  /status   - Get server status and available models
    POST /install  - Start installation (JSON body: {"models": [...], "hf_token": "...", ...})
    GET  /progress - Server-Sent Events stream for progress updates
    GET  /logs     - Get recent log messages
    POST /stop     - Stop current installation
    POST /pull     - Git pull latest changes
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Optional

# Configuration
PORT = 5111
COMFY_DIR = Path("/workspace/ComfyUI")
SETUP_DIR = Path(__file__).parent.resolve()
SOURCES_FILE = SETUP_DIR / "sources2.json"

# Global state
installation_status = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,
    "current_task": "",
    "error": None,
    "started_at": None,
}
log_buffer = []
log_queue = queue.Queue()
progress_clients = []
installation_thread = None
stop_flag = threading.Event()


def log(message: str, level: str = "info"):
    """Add a log message."""
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "level": level,
        "message": message
    }
    log_buffer.append(entry)
    if len(log_buffer) > 1000:
        log_buffer.pop(0)

    # Notify progress clients
    broadcast_event({"type": "log", "message": message, "level": level})
    print(f"[{entry['time']}] [{level.upper()}] {message}")


def broadcast_event(data: dict):
    """Send event to all connected SSE clients."""
    for q in progress_clients[:]:
        try:
            q.put_nowait(data)
        except queue.Full:
            pass


def update_progress(progress: int, task: str = None):
    """Update installation progress."""
    installation_status["progress"] = progress
    if task:
        installation_status["current_task"] = task

    broadcast_event({
        "progress": progress,
        "current_task": task or installation_status["current_task"]
    })


def load_sources() -> dict:
    """Load sources from JSON file."""
    if SOURCES_FILE.exists():
        with open(SOURCES_FILE, "r") as f:
            return json.load(f)
    return {"models": [], "custom_nodes": []}


def get_available_models() -> list:
    """Get list of available model categories."""
    sources = load_sources()
    categories = set()
    for model in sources.get("models", []):
        for cat in model.get("models", []):
            categories.add(cat)
    return sorted(list(categories))


def run_installation(models: list, hf_token: str = "", civitai_token: str = "", github_token: str = ""):
    """Run the installation process."""
    global installation_status

    try:
        installation_status["status"] = "running"
        installation_status["progress"] = 0
        installation_status["error"] = None
        installation_status["started_at"] = time.time()

        log(f"Starting installation for models: {', '.join(models)}")
        update_progress(5, "Loading sources...")

        # Check if setup_remote.py exists
        setup_script = SETUP_DIR / "setup_remote.py"
        if not setup_script.exists():
            raise FileNotFoundError(f"setup_remote.py not found at {setup_script}")

        # Build command
        cmd = [sys.executable, str(setup_script)]
        cmd.extend(["--models", ",".join(models)])

        if hf_token:
            cmd.extend(["--hf-token", hf_token])
        if civitai_token:
            cmd.extend(["--civitai-token", civitai_token])
        if github_token:
            cmd.extend(["--github-token", github_token])

        log(f"Running: {' '.join(cmd[:3])}...")
        update_progress(10, "Starting download process...")

        # Run the setup script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(SETUP_DIR)
        )

        # Stream output
        line_count = 0
        while True:
            if stop_flag.is_set():
                process.terminate()
                log("Installation cancelled by user", "warning")
                installation_status["status"] = "cancelled"
                return

            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            line = line.strip()
            if line:
                log(line)
                line_count += 1

                # Try to parse progress from output
                if "%" in line:
                    try:
                        # Look for percentage in line
                        import re
                        match = re.search(r'(\d+)%', line)
                        if match:
                            pct = int(match.group(1))
                            update_progress(10 + int(pct * 0.85), line[:60])
                    except:
                        pass
                elif "Downloading" in line or "Cloning" in line:
                    update_progress(installation_status["progress"], line[:60])
                elif "Downloaded" in line or "Installed" in line:
                    update_progress(min(95, installation_status["progress"] + 5), line[:60])

        return_code = process.wait()

        if return_code == 0:
            update_progress(100, "Installation complete!")
            installation_status["status"] = "completed"
            log("Installation completed successfully!")
            broadcast_event({"type": "status", "status": "completed"})
        else:
            installation_status["status"] = "error"
            installation_status["error"] = f"Process exited with code {return_code}"
            log(f"Installation failed with exit code {return_code}", "error")
            broadcast_event({"type": "status", "status": "error", "error": installation_status["error"]})

    except Exception as e:
        installation_status["status"] = "error"
        installation_status["error"] = str(e)
        log(f"Installation error: {e}", "error")
        broadcast_event({"type": "status", "status": "error", "error": str(e)})

    finally:
        stop_flag.clear()


class SetupHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler with CORS support."""

    def log_message(self, format, *args):
        """Override to use our logging."""
        log(f"HTTP: {args[0]}", "debug")

    def send_cors_headers(self):
        """Send CORS headers for all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Origin, User-Agent")
        self.send_header("Access-Control-Max-Age", "86400")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_json(self, data: dict, status: int = 200):
        """Send JSON response with CORS headers."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path == "/" or path == "/status":
            self.handle_status()
        elif path == "/progress":
            self.handle_progress_stream()
        elif path == "/logs":
            self.handle_logs()
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path

        if path == "/install":
            self.handle_install()
        elif path == "/stop":
            self.handle_stop()
        elif path == "/pull":
            self.handle_pull()
        else:
            self.send_json({"error": "Not found"}, 404)

    def handle_status(self):
        """Return server status."""
        sources = load_sources()
        available_models = get_available_models()

        response = {
            "status": "online",
            "installation": installation_status.copy(),
            "available": {
                "available_models": available_models,
                "total": {
                    "models": len(sources.get("models", [])),
                    "custom_nodes": len(sources.get("custom_nodes", []))
                }
            },
            "comfy_dir": str(COMFY_DIR),
            "comfy_exists": COMFY_DIR.exists()
        }
        self.send_json(response)

    def handle_install(self):
        """Start installation."""
        global installation_thread

        if installation_status["status"] == "running":
            self.send_json({"error": "Installation already running"}, 400)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        models = data.get("models", [])
        if not models:
            self.send_json({"error": "No models specified"}, 400)
            return

        hf_token = data.get("hf_token", "")
        civitai_token = data.get("civitai_token", "")
        github_token = data.get("github_token", "")

        # Start installation in background thread
        stop_flag.clear()
        installation_thread = threading.Thread(
            target=run_installation,
            args=(models, hf_token, civitai_token, github_token),
            daemon=True
        )
        installation_thread.start()

        self.send_json({"status": "started", "models": models})

    def handle_progress_stream(self):
        """Stream progress updates via Server-Sent Events."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_cors_headers()
        self.end_headers()

        # Create a queue for this client
        client_queue = queue.Queue(maxsize=100)
        progress_clients.append(client_queue)

        try:
            # Send initial status
            self.wfile.write(f"data: {json.dumps(installation_status)}\n\n".encode())
            self.wfile.flush()

            while True:
                try:
                    # Wait for event with timeout
                    event = client_queue.get(timeout=30)
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

                    # Check if installation is done
                    if event.get("type") == "status" and event.get("status") in ["completed", "error"]:
                        break

                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if client_queue in progress_clients:
                progress_clients.remove(client_queue)

    def handle_logs(self):
        """Return recent logs."""
        self.send_json({"logs": log_buffer[-100:]})

    def handle_stop(self):
        """Stop current installation."""
        if installation_status["status"] != "running":
            self.send_json({"error": "No installation running"}, 400)
            return

        stop_flag.set()
        self.send_json({"status": "stopping"})

    def handle_pull(self):
        """Git pull latest changes."""
        try:
            result = subprocess.run(
                ["git", "pull"],
                cwd=str(SETUP_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )
            self.send_json({
                "status": "success" if result.returncode == 0 else "error",
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            })
        except Exception as e:
            self.send_json({"status": "error", "error": str(e)}, 500)


class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    """HTTP server that handles requests in threads."""
    allow_reuse_address = True


def main():
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║          ComfyUI Setup Server v2.0                            ║
╠═══════════════════════════════════════════════════════════════╣
║  Port: {PORT}                                                   ║
║  Sources: {str(SOURCES_FILE)[:45]:<45} ║
║  ComfyUI: {str(COMFY_DIR)[:45]:<45} ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Check for sources file
    if not SOURCES_FILE.exists():
        print(f"[WARNING] Sources file not found: {SOURCES_FILE}")
        print("          The server will start but no models are configured.")

    # Start server
    server = ThreadedHTTPServer(("0.0.0.0", PORT), SetupHandler)
    print(f"Server listening on port {PORT}...")
    print(f"Access via RunPod proxy: https://YOUR_POD_ID-{PORT}.proxy.runpod.net/")
    print("\nEndpoints:")
    print("  GET  /status   - Server status and available models")
    print("  POST /install  - Start installation")
    print("  GET  /progress - SSE progress stream")
    print("  GET  /logs     - Recent log messages")
    print("  POST /stop     - Cancel installation")
    print("  POST /pull     - Git pull updates")
    print("\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
