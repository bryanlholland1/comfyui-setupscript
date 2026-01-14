#!/usr/bin/env python3
"""
RunPod ComfyUI Setup Script (Remote Version)
=============================================
Run this script directly on your RunPod instance to install models and custom nodes.

Usage:
    python setup_remote.py --models wan,qwen
    python setup_remote.py --models zit,ltx2
    python setup_remote.py --models all
    python setup_remote.py --list  # Show available models
    python setup_remote.py  # Interactive mode

With API tokens (for private repos/gated models):
    python setup_remote.py --models all --github-token ghp_xxx --hf-token hf_xxx

Or set environment variables:
    export GITHUB_TOKEN=ghp_xxx
    export HF_TOKEN=hf_xxx
    export CIVITAI_TOKEN=xxx
    python setup_remote.py --models all

GitHub repos in sources2.json:
    If a model's "source" is a GitHub URL (not ending in a file extension),
    it will be cloned to the "downloadDestination" path instead of downloaded.

Copy this file and sources2.json to your RunPod instance, then run it.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ANSI colors
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
RED = '\033[0;31m'
DIM = '\033[2m'
BOLD = '\033[1m'
NC = '\033[0m'  # No Color

# Paths on RunPod
COMFYUI_DIR = Path("/ComfyUI")
MODELS_DIR = COMFYUI_DIR / "models"
CUSTOM_NODES_DIR = COMFYUI_DIR / "custom_nodes"
WORKFLOWS_DIR = COMFYUI_DIR / "user" / "default" / "workflows"

# Model name mapping
MODEL_ALIASES = {
    "wan": "Wan2.2",
    "wan2.2": "Wan2.2",
    "qwen": "Qwen",
    "zit": "ZIT",
    "ltx2": "LTX2",
    "vibevoice": "VibeVoice",
}

ALL_MODELS = list(set(MODEL_ALIASES.values()))


def print_banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║           ComfyUI Model Setup (Remote)                   ║
╚══════════════════════════════════════════════════════════╝{NC}
""")


def load_sources(sources_path: str = None) -> dict:
    """Load sources from JSON file."""
    if sources_path is None:
        # Try current directory, then script directory
        candidates = [
            Path("sources2.json"),
            Path(__file__).parent / "sources2.json",
            Path.home() / "sources2.json",
        ]
        for p in candidates:
            if p.exists():
                sources_path = str(p)
                break

    if sources_path is None or not Path(sources_path).exists():
        print(f"{RED}Error: sources2.json not found!{NC}")
        print(f"Please copy sources2.json to the same directory as this script.")
        sys.exit(1)

    print(f"{DIM}Loading sources from: {sources_path}{NC}")
    with open(sources_path, "r") as f:
        return json.load(f)


def filter_items_by_models(items: list, selected_models: set, include_null: bool = True) -> list:
    """Filter items by associated model."""
    filtered = []
    for item in items:
        assoc = item.get("associatedModel")
        if assoc is None and include_null:
            filtered.append(item)
        elif assoc in selected_models:
            filtered.append(item)
    return filtered


def download_file(url: str, dest_path: Path, filename: str, hf_token: str = "", civitai_token: str = "") -> bool:
    """Download a file using wget."""
    full_path = dest_path / filename

    if full_path.exists():
        print(f"  {GREEN}✓{NC} {filename} {DIM}(exists){NC}")
        return True

    # Create directory if needed
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"  {CYAN}↓{NC} Downloading {filename}...")

    # Build wget command based on URL type
    cmd = ["wget", "-q", "--show-progress"]

    if "huggingface.co" in url and hf_token:
        cmd.extend(["--header", f"Authorization: Bearer {hf_token}"])
    elif "civitai.com" in url and civitai_token:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}token={civitai_token}"
        cmd.append("--content-disposition")

    cmd.extend(["-O", str(full_path), url])

    try:
        result = subprocess.run(cmd, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"    {GREEN}✓{NC} Downloaded")
            return True
        else:
            print(f"    {RED}✗{NC} Download failed")
            return False
    except subprocess.TimeoutExpired:
        print(f"    {YELLOW}!{NC} Download timed out")
        return False
    except Exception as e:
        print(f"    {RED}✗{NC} Error: {e}")
        return False


def is_github_url(url: str) -> bool:
    """Check if URL is a GitHub repository."""
    return "github.com" in url and not url.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".zip", ".tar.gz"))


def clone_repo(url: str, dest_path: Path, github_token: str = "") -> bool:
    """Clone a git repository."""
    if dest_path.exists():
        print(f"  {GREEN}✓{NC} {dest_path.name} {DIM}(exists){NC}")
        return True

    print(f"  {CYAN}↓{NC} Cloning {dest_path.name}...")

    # Insert GitHub token for private repos
    clone_url = url
    if github_token and "github.com" in url:
        # Convert https://github.com/user/repo to https://TOKEN@github.com/user/repo
        clone_url = url.replace("https://github.com", f"https://{github_token}@github.com")
        # Handle URLs without .git suffix
        if not clone_url.endswith(".git"):
            clone_url = clone_url.rstrip("/") + ".git"

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(dest_path)],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"    {GREEN}✓{NC} Installed")
            return True
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            # Don't leak token in error messages
            error_msg = error_msg.replace(github_token, "***") if github_token else error_msg
            print(f"    {RED}✗{NC} Clone failed: {error_msg[:100]}")
            return False
    except Exception as e:
        print(f"    {RED}✗{NC} Error: {e}")
        return False


def setup_models(sources: dict, selected_models: set, hf_token: str = "", civitai_token: str = "", github_token: str = ""):
    """Download all models for selected model types."""
    filtered_models = filter_items_by_models(sources.get("models", []), selected_models)

    print(f"\n{BOLD}{'━' * 60}{NC}")
    print(f"{CYAN}↓{NC} {BOLD}Downloading Models{NC} ({len(filtered_models)} items)")
    print(f"{BOLD}{'━' * 60}{NC}\n")

    downloaded = 0
    cloned = 0
    skipped = 0

    for item in filtered_models:
        source = item.get("source", "")
        if source == "NEEDS_SOURCE" or not source:
            skipped += 1
            continue

        filename = item.get("fileName", "")
        dest = item.get("downloadDestination", "").lstrip("/")

        # Handle destination path
        if dest.startswith("ComfyUI/"):
            dest = dest[8:]

        dest_path = COMFYUI_DIR / dest

        # Check if this is a GitHub repo to clone
        if is_github_url(source):
            # For repos, the destination is the full path including repo name
            if clone_repo(source, dest_path, github_token):
                cloned += 1
        else:
            # Regular file download
            if download_file(source, dest_path, filename, hf_token, civitai_token):
                downloaded += 1

    result_parts = []
    if downloaded > 0:
        result_parts.append(f"Downloaded {downloaded} files")
    if cloned > 0:
        result_parts.append(f"Cloned {cloned} repos")

    print(f"\n{GREEN}{', '.join(result_parts) if result_parts else 'No items processed'}{NC} {DIM}({skipped} skipped - no source){NC}")


def setup_custom_nodes(sources: dict, selected_models: set, github_token: str = ""):
    """Clone all custom nodes for selected model types."""
    filtered_nodes = filter_items_by_models(sources.get("custom_nodes", []), selected_models)

    print(f"\n{BOLD}{'━' * 60}{NC}")
    print(f"{CYAN}↓{NC} {BOLD}Installing Custom Nodes{NC} ({len(filtered_nodes)} nodes)")
    print(f"{BOLD}{'━' * 60}{NC}\n")

    installed = 0
    skipped = 0

    for node in filtered_nodes:
        source = node.get("source", "")
        if source == "NEEDS_SOURCE" or not source:
            skipped += 1
            continue

        dest = node.get("downloadDestination", "").lstrip("/")

        if dest.startswith("ComfyUI/"):
            dest = dest[8:]

        dest_path = COMFYUI_DIR / dest

        if clone_repo(source, dest_path, github_token):
            installed += 1

    print(f"\n{GREEN}Installed {installed} custom nodes{NC} {DIM}({skipped} skipped - no source){NC}")


def create_directories():
    """Create necessary directories."""
    dirs = [
        MODELS_DIR / "diffusion_models",
        MODELS_DIR / "diffusion_models" / "WAN 2.2",
        MODELS_DIR / "diffusion_models" / "wan2.2",
        MODELS_DIR / "text_encoders",
        MODELS_DIR / "vae",
        MODELS_DIR / "loras",
        MODELS_DIR / "loras" / "Wan",
        MODELS_DIR / "loras" / "SVI",
        MODELS_DIR / "loras" / "lightx2v",
        MODELS_DIR / "loras" / "LTX",
        MODELS_DIR / "checkpoints",
        MODELS_DIR / "upscale_models",
        MODELS_DIR / "latent_upscale_models",
        MODELS_DIR / "clip_vision",
        WORKFLOWS_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def list_models(sources: dict):
    """List available models and their contents."""
    print(f"\n{BOLD}Available Models:{NC}\n")

    for model_name in sorted(ALL_MODELS):
        model_items = [m for m in sources.get("models", []) if m.get("associatedModel") == model_name]
        node_items = [n for n in sources.get("custom_nodes", []) if n.get("associatedModel") == model_name]

        print(f"  {CYAN}{model_name}{NC}")
        print(f"    Models: {len(model_items)}, Custom Nodes: {len(node_items)}")

    print(f"\n{DIM}Use: python setup_remote.py --models wan,zit,ltx2{NC}")
    print(f"{DIM}Or:  python setup_remote.py --models all{NC}\n")


def interactive_selection(sources: dict) -> set:
    """Simple interactive model selection."""
    print(f"\n{BOLD}Select models to install:{NC}\n")

    models = sorted(ALL_MODELS)
    for i, model in enumerate(models, 1):
        model_count = sum(1 for m in sources.get("models", []) if m.get("associatedModel") == model)
        node_count = sum(1 for n in sources.get("custom_nodes", []) if n.get("associatedModel") == model)
        print(f"  {i}. {model} ({model_count} models, {node_count} nodes)")

    print(f"  {len(models) + 1}. ALL models")
    print(f"  0. Cancel")

    print(f"\n{CYAN}Enter numbers separated by commas (e.g., 1,3,4):{NC} ", end="")

    try:
        choice = input().strip()
    except EOFError:
        return set()

    if not choice or choice == "0":
        return set()

    selected = set()
    for num in choice.split(","):
        try:
            n = int(num.strip())
            if n == len(models) + 1:
                return set(ALL_MODELS)
            elif 1 <= n <= len(models):
                selected.add(models[n - 1])
        except ValueError:
            pass

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Set up ComfyUI models and custom nodes on RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--models", "-m", help="Comma-separated list of models (wan,zit,ltx2,qwen,vibevoice,all)")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--sources", "-s", help="Path to sources2.json file")
    parser.add_argument("--hf-token", help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--civitai-token", help="CivitAI API token (or set CIVITAI_TOKEN env var)")
    parser.add_argument("--github-token", help="GitHub token for private repos (or set GITHUB_TOKEN env var)")
    parser.add_argument("--skip-models", action="store_true", help="Skip downloading models")
    parser.add_argument("--skip-nodes", action="store_true", help="Skip installing custom nodes")

    args = parser.parse_args()

    print_banner()

    # Load sources
    sources = load_sources(args.sources)
    print(f"{GREEN}✓{NC} Loaded {len(sources.get('models', []))} models, {len(sources.get('custom_nodes', []))} custom nodes\n")

    # List mode
    if args.list:
        list_models(sources)
        return

    # Get API tokens from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    civitai_token = args.civitai_token or os.environ.get("CIVITAI_TOKEN", "")
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN", "")

    # Model selection
    if args.models:
        models_input = [m.strip().lower() for m in args.models.split(",")]
        selected_models = set()

        for m in models_input:
            if m == "all":
                selected_models = set(ALL_MODELS)
                break
            elif m in MODEL_ALIASES:
                selected_models.add(MODEL_ALIASES[m])
            else:
                print(f"{YELLOW}Warning: Unknown model '{m}'{NC}")

        if not selected_models:
            print(f"{RED}No valid models specified{NC}")
            sys.exit(1)
    else:
        selected_models = interactive_selection(sources)
        if not selected_models:
            print(f"{YELLOW}No models selected. Exiting.{NC}")
            return

    print(f"\n{BOLD}Selected models:{NC} {CYAN}{', '.join(sorted(selected_models))}{NC}\n")

    # Create directories
    print(f"{DIM}Creating directories...{NC}")
    create_directories()
    print(f"{GREEN}✓{NC} Directories ready\n")

    # Download models
    if not args.skip_models:
        setup_models(sources, selected_models, hf_token, civitai_token, github_token)

    # Install custom nodes
    if not args.skip_nodes:
        setup_custom_nodes(sources, selected_models, github_token)

    # Done
    print(f"\n{BOLD}{GREEN}╔══════════════════════════════════════════════════════════╗{NC}")
    print(f"{BOLD}{GREEN}║              ✓ Installation Complete!                    ║{NC}")
    print(f"{BOLD}{GREEN}╚══════════════════════════════════════════════════════════╝{NC}")
    print(f"\n{CYAN}Models:{NC} {', '.join(sorted(selected_models))}")
    print(f"{CYAN}Workflows:{NC} {WORKFLOWS_DIR}")
    print(f"\n{DIM}You may need to restart ComfyUI to load new custom nodes.{NC}\n")


if __name__ == "__main__":
    main()
