#!/usr/bin/env python3
"""
scripts/setup.py — one-shot environment bootstrapper
=====================================================
Downloads all required GGUF models and builds turbovec from source.

Run::

    python scripts/setup.py            # download all + build turbovec
    python scripts/setup.py --models   # models only
    python scripts/setup.py --build    # turbovec build only
    python scripts/setup.py --list     # list available models
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ── Model registry ────────────────────────────────────────────────────────────
# Format: { key: { "url": "...", "path": "models/...", "desc": "..." } }
# URLs point to HuggingFace or compatible hosts.
# Replace with your preferred mirrors if needed.

MODELS = {
    # ── Embedding ─────────────────────────────────────────────────────────────
    "gemma-embedding-270m": {
        "desc": "Gemma Embedding 270M — Q4_K_M (text embedding, dim=2048)",
        "url":  (
            "https://huggingface.co/sabafallah/embeddinggemma-300m-Q4_K_M-GGUF/resolve/main/"
            "embeddinggemma-300m-q4_k_m.gguf"
            # you can  change  with another models  gemma 
        ),
        "path": "models/embeddinggemma-300m-q4_k_m.gguf",
        "required": True,
    },
    # ── Generation models ─────────────────────────────────────────────────────
    "qwen-0.5b": {
        "desc": "Qwen 0.5B — Q4_K_M (fast generation, ~300MB)",
        "url":  (
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
            "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "required": False,
    },
    "deepseek-r1-1.5B": {
        "desc": "DeepSeek-R1 1.5B — Q4_K_M (reasoning, ~900MB)",
        "url":  (
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/"
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
        ),
        "path": "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "required": False,
    },
    "deepseek-1.3B": {
        "desc": "DeepSeek 1.3B — Q4_K_M (reasoning, ~800MB)",
        "url":  (
            "https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/resolve/main/"
            "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
        ),
        "path": "models/deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        "required": False,
    },
    # ── Vision-Language models ────────────────────────────────────────────────
    "smolvlm-135m": {
        "desc": "SmolVLM 135M — Q4_K_M (ultra-light LLM, ~90MB)",
        "url":  (
            "https://huggingface.co/ggml-org/SmolVLM-135M-Instruct-GGUF/resolve/main/"
            "SmolVLM-135M-Instruct-Q4_K_M.gguf"
        ),
        "path": "models/smolvlm-135m-Q4_K_M.gguf",
        "required": False,
    },
    "smolvlm-256m": {
        "desc": "SmolVLM 256M — Q4_K_M (balanced VLM, ~160MB)",
        "url":  (
            "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/"
            "SmolVLM-256M-Instruct-Q4_K_M.gguf"
        ),
        "path": "models/smolvlm-256m-Q4_K_M.gguf",
        "required": False,
    },
    "smolvlm-500m": {
        "desc": "SmolVLM 500M — Q4_K_M (best VLM quality, ~320MB)",
        "url":  (
            "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/"
            "SmolVLM-500M-Instruct-Q4_K_M.gguf"
        ),
        "path": "models/smolvlm-500m-Q4_K_M.gguf",
        "required": False,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(count, block_size, total):
    if total > 0:
        pct = min(100, count * block_size * 100 // total)
        mb = count * block_size / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


def download_model(key: str, force: bool = False) -> bool:
    info = MODELS[key]
    dest = Path(info["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  ✓ Already exists: {dest}")
        return True

    print(f"\n⬇ Downloading: {info['desc']}")
    print(f"  URL:  {info['url']}")
    print(f"  Dest: {dest}")

    try:
        urllib.request.urlretrieve(info["url"], dest, reporthook=_progress)
        print(f"\n  ✓ Saved: {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return True
    except Exception as exc:
        print(f"\n  ✗ Failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def build_turbovec(zip_path: str = "turbovec-main.zip") -> bool:
    """Build turbovec from the source zip and install it."""
    import zipfile, tempfile, shutil

    if not os.path.exists(zip_path):
        print(f"⚠ turbovec source not found at {zip_path}")
        print("  Download from: https://github.com/RyanCodrai/turbovec")
        return False

    print("\n🔨 Building turbovec from source …")

    # Check Rust
    if subprocess.run(["rustc", "--version"], capture_output=True).returncode != 0:
        print("  ✗ Rust not installed.  Install from: https://rustup.rs/")
        return False

    # Check maturin
    if subprocess.run([sys.executable, "-m", "maturin", "--version"],
                      capture_output=True).returncode != 0:
        print("  Installing maturin …")
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        build_dir = Path(tmp) / "turbovec-main" / "turbovec-python"
        if not build_dir.exists():
            print(f"  ✗ Expected turbovec-python/ inside zip, not found.")
            return False

        print("  Running: maturin develop --release")
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            cwd=build_dir,
        )
        if result.returncode != 0:
            print("  ✗ Build failed.")
            return False

    print("  ✓ turbovec installed successfully.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboRag environment setup")
    parser.add_argument("--models",   action="store_true", help="Download models only")
    parser.add_argument("--build",    action="store_true", help="Build turbovec only")
    parser.add_argument("--list",     action="store_true", help="List available models")
    parser.add_argument("--force",    action="store_true", help="Re-download existing files")
    parser.add_argument("--keys",     nargs="*",           help="Specific model keys to download")
    parser.add_argument("--zip",      default="turbovec-main.zip", help="Path to turbovec zip")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for k, v in MODELS.items():
            mark = "✓" if Path(v["path"]).exists() else " "
            req  = "[required]" if v["required"] else ""
            print(f"  [{mark}] {k:30s} — {v['desc']} {req}")
        return

    do_models = args.models or (not args.build)
    do_build  = args.build  or (not args.models)

    if do_models:
        keys = args.keys or list(MODELS.keys())
        print(f"\n📦 Downloading {len(keys)} model(s) …")
        ok = all(download_model(k, force=args.force) for k in keys if k in MODELS)
        if ok:
            print("\n✓ All models ready.")
        else:
            print("\n⚠ Some downloads failed — check URLs above.")

    if do_build:
        ok = build_turbovec(args.zip)
        if ok:
            print("\n✓ turbovec ready.")
        else:
            print("\n⚠ turbovec build failed — see messages above.")

    print("\n📋 Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. pip install -e .   (install turborag)")
    print("  3. Download a .zim file from https://download.kiwix.org/zim/wikipedia/")
    print("  4. python main.py index --zim data/wikipedia_en_mini.zim")


if __name__ == "__main__":
    main()
