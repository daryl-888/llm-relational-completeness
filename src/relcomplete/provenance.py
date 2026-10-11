"""Run manifests: everything needed to say exactly what produced a results file."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .prompts import ROWWISE_PARAMETRIC, SYSTEM, VARIANTS

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["openai", "faker", "pandas", "periodictable", "rank-bm25", "matplotlib"]


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _cmd(args: list[str]) -> str | None:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return (p.stdout.strip() or None) if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def manifest(bench_dir: Path, question_ids: list[str], model: str, variant: str, repeat: int,
             argv: list[str], max_tokens: int) -> dict:
    code = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "variant": variant, "repeat": repeat, "max_tokens": max_tokens,
        "command": [sys.executable, *argv],
        "n_questions": len(question_ids),
        "question_ids_sha256": hashlib.sha256("\n".join(question_ids).encode()).hexdigest(),
        "question_ids": question_ids,
        "benchmark": {"path": str(bench_dir),
                      "sha256": sha256_files([Path(bench_dir).resolve() / "questions.jsonl",
                                              Path(bench_dir).resolve() / "tables.jsonl"])},
        "code_sha256": sha256_files([p for p in code if p.is_file()]),
        "git": {"commit": _cmd(["git", "rev-parse", "--verify", "HEAD"]),
                "dirty": bool(_cmd(["git", "status", "--porcelain"]))},
        "prompts": {"system": SYSTEM, "variants": VARIANTS, "rowwise_parametric": ROWWISE_PARAMETRIC},
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "claude_cli_version": _cmd(["claude", "--version"]) if model.startswith("claude-cli/") else None,
    }


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None
