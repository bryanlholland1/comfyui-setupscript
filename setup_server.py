#!/usr/bin/env python3
"""
ComfyUI Setup Server - Runs on RunPod instance
===============================================
A lightweight web service that allows remote configuration and installation
of ComfyUI models and custom nodes.

Usage:
    python setup_server.py

One-liner for RunPod:
    git clone https://github.com/YOUR_USER/YOUR_REPO.git /root/comfy-setup && python /root/comfy-setup/setup_server.py

Endpoints:
    GET  /                - Health check
    GET  /status          - Get available models and current status
    POST /install         - Start installation (JSON body: {"models": ["Wan2.2", "ZIT"], "tokens": {...}})
    GET  /progress        - Server-Sent Events stream of installation progress
    GET  /logs            - Get current log buffer
    POST /stop            - Stop current installation
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import re

PORT = 5111
SCRIPT_DIR = Path(__file__).parent
SOURCES_FILE = SCRIPT_DIR / "sources2.json"
SETUP_SCRIPT = SCRIPT_DIR / "setup_remote.py"

# Global state
installation_state = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,
    "current_task": "",
    "started_at": None,
    "completed_at": None,
    "error": None,
}

log_buffer = []
log_buffer_lock = threading.Lock()
progress_queue = queue.Queue()
current_process = None
process_lock = threading.Lock()

# ANSI color codes for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
RED = '\033[0;31m'
BOLD = '\033[1m'
NC = '\033[0m'


def load_sources():
    """Load sources2.json and return model info."""
    if not SOURCES_FILE.exists():
        return {"models": [], "custom_nodes": []}

    with open(SOURCES_FILE) as f:
        return json.load(f)


def get_available_models():
    """Get list of available model types from sources."""
    sources = load_sources()

    # Get unique associated models
    models = set()
    for item in sources.get("models", []):
        if item.get("associatedModel"):
            models.add(item.get("associatedModel"))

    # Count items per model
    model_info = {}
    for model in models:
        model_items = [m for m in sources.get("models", []) if m.get("associatedModel") == model]
        node_items = [n for n in sources.get("custom_nodes", []) if n.get("associatedModel") == model]
        model_info[model] = {
            "models": len(model_items),
            "custom_nodes": len(node_items),
        }

    # Add shared items (no associatedModel)
    shared_models = [m for m in sources.get("models", []) if not m.get("associatedModel")]
    shared_nodes = [n for n in sources.get("custom_nodes", []) if not n.get("associatedModel")]

    return {
        "available_models": sorted(list(models)),
        "model_details": model_info,
        "shared": {
            "models": len(shared_models),
            "custom_nodes": len(shared_nodes),
        },
        "total": {
            "models": len(sources.get("models", [])),
            "custom_nodes": len(sources.get("custom_nodes", [])),
        }
    }


def add_log(message):
    """Add a message to the log buffer and progress queue."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"

    with log_buffer_lock:
        log_buffer.append(entry)
        # Keep last 1000 lines
        if len(log_buffer) > 1000:
            log_buffer.pop(0)

    # Add to progress queue for SSE
    progress_queue.put({"type": "log", "message": message})


def run_installation(models, tokens):
    """Run the installation in a background thread."""
    global installation_state, current_process

    installation_state["status"] = "running"
    installation_state["progress"] = 0
    installation_state["current_task"] = "Starting installation..."
    installation_state["started_at"] = time.time()
    installation_state["completed_at"] = None
    installation_state["error"] = None

    add_log(f"Starting installation for models: {', '.join(models)}")
    progress_queue.put({"type": "status", "status": "running"})

    try:
        # Build command
        cmd = [sys.executable, str(SETUP_SCRIPT), "--models", ",".join(models)]

        if tokens.get("hf_token"):
            cmd.extend(["--hf-token", tokens["hf_token"]])
        if tokens.get("civitai_token"):
            cmd.extend(["--civitai-token", tokens["civitai_token"]])
        if tokens.get("github_token"):
            cmd.extend(["--github-token", tokens["github_token"]])

        add_log(f"Running: python setup_remote.py --models {','.join(models)}")

        # Run process and capture output
        with process_lock:
            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(SCRIPT_DIR)
            )

        # Stream output
        total_items = 0
        completed_items = 0

        for line in current_process.stdout:
            line = line.rstrip()
            if not line:
                continue

            # Strip ANSI codes for logging
            clean_line = re.sub(r'\033\[[0-9;]*m', '', line)
            add_log(clean_line)

            # Parse progress indicators
            if "Downloading Models" in line:
                match = re.search(r'\((\d+)', line)
                if match:
                    total_items = int(match.group(1))
                installation_state["current_task"] = "Downloading models..."
            elif "Installing Custom Nodes" in line:
                match = re.search(r'\((\d+)', line)
                if match:
                    total_items = int(match.group(1))
                installation_state["current_task"] = "Installing custom nodes..."
            elif "✓" in clean_line or "Downloaded" in clean_line or "Installed" in clean_line:
                completed_items += 1
                if total_items > 0:
                    installation_state["progress"] = min(95, int(completed_items / total_items * 90))
            elif "↓" in clean_line:
                # Extract filename being downloaded
                match = re.search(r'↓\s*(?:Downloading\s+)?(.+?)(?:\.\.\.)?$', clean_line)
                if match:
                    installation_state["current_task"] = f"Downloading {match.group(1).strip()}..."

            progress_queue.put({
                "type": "progress",
                "progress": installation_state["progress"],
                "current_task": installation_state["current_task"]
            })

        current_process.wait()

        with process_lock:
            current_process = None

        if installation_state["status"] == "running":  # Not cancelled
            installation_state["status"] = "completed"
            installation_state["progress"] = 100
            installation_state["current_task"] = "Installation complete!"
            installation_state["completed_at"] = time.time()
            add_log("Installation completed successfully!")
            progress_queue.put({"type": "status", "status": "completed"})

    except Exception as e:
        installation_state["status"] = "error"
        installation_state["error"] = str(e)
        add_log(f"Error: {e}")
        progress_queue.put({"type": "error", "error": str(e)})

        with process_lock:
            current_process = None


class SetupHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the setup server."""

    def log_message(self, format, *args):
        # Custom logging
        print(f"{CYAN}[HTTP]{NC} {args[0]}")

    def send_json(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path

        if path == "/" or path == "/health":
            self.send_json({
                "status": "ok",
                "service": "ComfyUI Setup Server",
                "version": "1.0.0",
            })

        elif path == "/status":
            models_info = get_available_models()
            self.send_json({
                "installation": installation_state,
                "available": models_info,
            })

        elif path == "/logs":
            with log_buffer_lock:
                logs = list(log_buffer)
            self.send_json({"logs": logs})

        elif path == "/progress":
            # Server-Sent Events stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                # Send current state first
                self.wfile.write(f"data: {json.dumps(installation_state)}\n\n".encode())
                self.wfile.flush()

                # Stream updates
                while True:
                    try:
                        update = progress_queue.get(timeout=1)
                        self.wfile.write(f"data: {json.dumps(update)}\n\n".encode())
                        self.wfile.flush()

                        # Stop streaming if installation completed or errored
                        if update.get("type") == "status" and update.get("status") in ["completed", "error"]:
                            break
                    except queue.Empty:
                        # Send keepalive
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()

                        # Check if installation finished
                        if installation_state["status"] in ["completed", "error", "idle"]:
                            break
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        path = urlparse(self.path).path

        if path == "/install":
            # Check if already running
            if installation_state["status"] == "running":
                self.send_json({"error": "Installation already in progress"}, 409)
                return

            # Parse request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON"}, 400)
                return

            models = data.get("models", [])
            if not models:
                self.send_json({"error": "No models specified"}, 400)
                return

            tokens = {
                "hf_token": data.get("hf_token", ""),
                "civitai_token": data.get("civitai_token", ""),
                "github_token": data.get("github_token", ""),
            }

            # Clear log buffer
            with log_buffer_lock:
                log_buffer.clear()

            # Clear progress queue
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except queue.Empty:
                    break

            # Start installation in background thread
            thread = threading.Thread(target=run_installation, args=(models, tokens))
            thread.daemon = True
            thread.start()

            self.send_json({
                "status": "started",
                "models": models,
                "message": "Installation started. Use /progress for updates."
            })

        elif path == "/stop":
            global current_process

            with process_lock:
                if current_process:
                    current_process.terminate()
                    current_process = None

            installation_state["status"] = "idle"
            installation_state["current_task"] = "Cancelled"
            add_log("Installation cancelled by user")

            self.send_json({"status": "stopped"})

        elif path == "/pull":
            # Pull latest from git
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=str(SCRIPT_DIR),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                self.send_json({
                    "status": "ok",
                    "output": result.stdout + result.stderr
                })
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        else:
            self.send_json({"error": "Not found"}, 404)


def main():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║           ComfyUI Setup Server                           ║
╚══════════════════════════════════════════════════════════╝{NC}

  {GREEN}►{NC} Server running on port {BOLD}{PORT}{NC}
  {GREEN}►{NC} Sources file: {SOURCES_FILE}

  {BOLD}Endpoints:{NC}
    GET  /status   - Get available models and status
    POST /install  - Start installation
    GET  /progress - Stream progress (SSE)
    GET  /logs     - Get log buffer
    POST /stop     - Cancel installation
    POST /pull     - Git pull latest changes

  {YELLOW}Waiting for connections...{NC}
  Press Ctrl+C to stop
""")

    # Check if sources file exists
    if not SOURCES_FILE.exists():
        print(f"{RED}Warning: {SOURCES_FILE} not found!{NC}")

    if not SETUP_SCRIPT.exists():
        print(f"{RED}Warning: {SETUP_SCRIPT} not found!{NC}")

    server = HTTPServer(("0.0.0.0", PORT), SetupHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Server stopped{NC}")
        server.shutdown()


if __name__ == "__main__":
    main()
