"""Guardie che impediscono al progetto di danneggiare la macchina che lo ospita.

Principio I della costituzione: i flip di questo progetto sono aritmetica in memoria,
mai eventi fisici. Cio che puo davvero danneggiare l'host e prosaico — un disco
riempito, un file di modello sovrascritto — ed e qui che si difende.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

BYTES_PER_GIB = 1024**3
DEFAULT_MIN_FREE_GIB = 10.0

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class HostSafetyError(RuntimeError):
    """L'involucro di salvaguardia e stato violato: l'operazione va fermata."""


class InsufficientDiskSpaceError(HostSafetyError):
    """Spazio libero sotto la soglia richiesta."""


class FileIntegrityError(HostSafetyError):
    """Un file dichiarato immutabile e cambiato."""


def free_gib(path: Path | str) -> float:
    """Spazio libero, in GiB, sul filesystem che contiene `path`."""
    return shutil.disk_usage(Path(path)).free / BYTES_PER_GIB


def require_free_space(
    path: Path | str, min_free_gib: float = DEFAULT_MIN_FREE_GIB
) -> float:
    """Verifica lo spazio libero prima di un'operazione che scrive.

    Restituisce i GiB disponibili; solleva `InsufficientDiskSpaceError` sotto soglia.
    """
    available = free_gib(path)
    if available < min_free_gib:
        raise InsufficientDiskSpaceError(
            f"{available:.1f} GiB liberi su {path}, soglia {min_free_gib:.1f} GiB"
        )
    return available


def sha256_file(path: Path | str) -> str:
    """Digest esadecimale SHA-256 del contenuto del file."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_readonly(path: Path | str) -> None:
    """Toglie i permessi di scrittura al file, seguendo gli eventuali symlink."""
    target = Path(path).resolve()
    target.chmod(target.stat().st_mode & ~WRITE_BITS)


def is_readonly(path: Path | str) -> bool:
    return not Path(path).resolve().stat().st_mode & WRITE_BITS


@contextmanager
def immutable(paths: Iterable[Path | str]) -> Iterator[dict[str, str]]:
    """Contesto che pretende che i file indicati non cambino di un solo bit.

    Registra i digest all'ingresso, li riverifica all'uscita, solleva
    `FileIntegrityError` alla prima differenza.
    """
    recorded = {str(Path(p).resolve()): sha256_file(p) for p in paths}
    try:
        yield recorded
    finally:
        # La verifica gira anche quando il corpo e fallito: un file corrotto durante
        # un errore e un fatto piu grave dell'errore stesso, e Python conserva
        # comunque l'eccezione originale nella catena.
        for filename, expected in recorded.items():
            actual = sha256_file(filename)
            if actual != expected:
                raise FileIntegrityError(
                    f"{filename} e cambiato: atteso {expected[:16]}…, "
                    f"trovato {actual[:16]}…"
                )
