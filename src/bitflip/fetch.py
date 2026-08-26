"""Acquiring the artifacts, under the guards of Principle I.

Every download is anchored to a precise repository revision: an experiment that does
not know which bytes it measured is not reproducible. Downloaded files are made
read-only immediately, and their digests go into the versioned manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bitflip.guard import make_readonly, require_free_space, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = PROJECT_ROOT / "results" / "models-manifest.json"

MIN_FREE_GIB_FOR_DOWNLOAD = 10.0

WEIGHT_PATTERNS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class Artifact:
    """A downloadable artifact, with the role it plays in the experiment."""

    key: str
    repo_id: str
    revision: str
    role: str
    primary: str
    patterns: tuple[str, ...] = WEIGHT_PATTERNS

    @property
    def local_dir(self) -> Path:
        return MODELS_DIR / self.key

    @property
    def primary_path(self) -> Path:
        return self.local_dir / self.primary


BASE = Artifact(
    key="base",
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    role="subject: the aligned model in bf16",
    primary="model.safetensors",
)

ABLITERATED = Artifact(
    key="abliterated",
    repo_id="huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3",
    revision="3dee99dac7c99318ed2b4e9932bfbbac060fb024",
    role="positive control: ablation only of the same base, no fine-tuning",
    primary="model.safetensors",
)

QUANTIZED = Artifact(
    key="gguf-q4-k-m",
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    revision="9217f5db79a29953eb74d5343926648285ec7e67",
    role="subject: the same model quantized to 4 bits",
    primary="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    patterns=("qwen2.5-0.5b-instruct-q4_k_m.gguf",),
)

ARTIFACTS = {artifact.key: artifact for artifact in (BASE, ABLITERATED, QUANTIZED)}


def freeze_files(root: Path) -> int:
    """Make every file under `root` read-only, leaving directories writable."""
    frozen = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            make_readonly(path)
            frozen += 1
    return frozen


def fetch(artifact: Artifact) -> Path:
    """Download the artifact at the pinned revision and freeze it read-only."""
    from huggingface_hub import snapshot_download

    require_free_space(PROJECT_ROOT, MIN_FREE_GIB_FOR_DOWNLOAD)
    MODELS_DIR.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=artifact.repo_id,
        revision=artifact.revision,
        local_dir=artifact.local_dir,
        allow_patterns=list(artifact.patterns),
    )
    freeze_files(artifact.local_dir)
    return artifact.primary_path


def describe(artifact: Artifact) -> dict[str, object]:
    """A manifest row: exactly what was measured."""
    path = artifact.primary_path
    return {
        "key": artifact.key,
        "repo_id": artifact.repo_id,
        "revision": artifact.revision,
        "role": artifact.role,
        "primary": artifact.primary,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    entries = []
    for artifact in ARTIFACTS.values():
        print(f"→ {artifact.repo_id}@{artifact.revision[:7]} ({artifact.role})")
        fetch(artifact)
        entry = describe(artifact)
        entries.append(entry)
        print(f"  {entry['bytes'] / 1e6:.0f} MB · sha256 {entry['sha256'][:16]}…")

    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest -> {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
