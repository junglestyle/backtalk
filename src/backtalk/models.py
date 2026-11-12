"""The models backtalk runs, each pinned to a Hugging Face commit and checked by hash.

They download into the cache directory on first start, as plain files (not
the Hugging Face cache's links, which onnxruntime won't follow to an
encoder's external weights), and load once, shared by every session.
"""

import hashlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime

from backtalk.audio import MODEL_RATE, Samples
from backtalk.vad import FRAME


@dataclass(frozen=True)
class Pinned:
    repo: str
    revision: str
    # File name to sha256: the files backtalk uses, as that commit has them.
    files: dict[str, str]


# NVIDIA's Parakeet TDT 0.6B v3 (CC BY 4.0), exported to ONNX for onnx-asr.
PARAKEET = Pinned(
    repo="istupakov/parakeet-tdt-0.6b-v3-onnx",
    revision="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
    files={
        "config.json": "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
        "vocab.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
        "encoder-model.onnx": "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
        "encoder-model.onnx.data": "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
        "decoder_joint-model.onnx": "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1",
    },
)
PARAKEET_NAME = "nemo-parakeet-tdt-0.6b-v3"
# Silero VAD v5 (MIT).
SILERO = Pinned(
    repo="istupakov/silero-vad-onnx",
    revision="b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6",
    files={"silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"},
)
# What session.created names as the session's model and transcription model.
MODEL = "parakeet-tdt-0.6b-v3"
_CONTEXT = 64
_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"


def fetch(pinned: Pinned, cache: Path) -> Path:
    """The directory holding the pinned files, downloaded unless the cache
    already holds exactly them."""
    from huggingface_hub import snapshot_download

    directory = cache / pinned.repo
    if _holds(directory, pinned):
        return directory
    print(f"downloading {pinned.repo}@{pinned.revision} into {directory}", file=sys.stderr)
    snapshot_download(
        pinned.repo,
        revision=pinned.revision,
        local_dir=directory,
        allow_patterns=list(pinned.files),
        force_download=True,
    )
    if not _holds(directory, pinned):
        raise SystemExit(f"{pinned.repo}@{pinned.revision} downloaded with other contents than backtalk pins")
    return directory


def _holds(directory: Path, pinned: Pinned) -> bool:
    return all(_sha256(directory / name) == digest for name, digest in pinned.files.items())


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


class Parakeet:
    """Speech to text on the GPU with onnxruntime's CUDA build, else (on a Mac) on the CPU."""

    def __init__(self, cache: Path) -> None:
        import onnx_asr

        providers = _providers(cache)
        started = time.monotonic()
        self._model = onnx_asr.load_model(PARAKEET_NAME, fetch(PARAKEET, cache), providers=providers)
        # Sessions share the model; one transcription runs at a time.
        self._lock = threading.Lock()
        # The first run builds CUDA kernels: done here, not on someone's first turn.
        self.transcribe(np.zeros(MODEL_RATE, dtype=np.float32))
        print(f"loaded {MODEL} on {providers[0]} in {time.monotonic() - started:.1f} s", file=sys.stderr)

    def transcribe(self, audio: Samples) -> str:
        with self._lock:
            text: str = self._model.recognize(audio, sample_rate=MODEL_RATE)
        return text.strip()


def _providers(cache: Path) -> list[str]:
    """CUDA where onnxruntime is built with it, else the CPU.

    A CUDA build that can't reach a GPU falls back to the CPU with no more than
    a logged error, so a small model (Silero's) is loaded on CUDA first to see:
    a GPU host's transcription that quietly ran on its CPU would be too slow
    to talk to."""
    if _CUDA not in onnxruntime.get_available_providers():
        return [_CPU]
    probe = onnxruntime.InferenceSession(
        str(fetch(SILERO, cache) / "silero_vad.onnx"), providers=[_CUDA, _CPU]
    )
    if probe.get_providers()[0] != _CUDA:
        raise SystemExit(
            "onnxruntime can't use the GPU (its error is above): is the GPU passed into the container?"
        )
    return [_CUDA, _CPU]


class Silero:
    """Speech probabilities for 32 ms frames, on the CPU: per frame, the GPU's
    launch costs more than the model."""

    def __init__(self, cache: Path) -> None:
        model = fetch(SILERO, cache) / "silero_vad.onnx"
        self._session = onnxruntime.InferenceSession(str(model), providers=[_CPU])

    def scorer(self) -> SileroScorer:
        return SileroScorer(self._session)


class SileroScorer:
    """One session's stream through Silero: its recurrent state and the last
    frame's tail, which the next frame is scored with."""

    def __init__(self, session: onnxruntime.InferenceSession) -> None:
        self._session = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def score(self, frame: Samples) -> float:
        if len(frame) != FRAME:
            raise ValueError(f"silero scores frames of {FRAME} samples, not {len(frame)}")
        window = np.concatenate([self._context, frame])[np.newaxis, :]
        output, self._state = self._session.run(
            ["output", "stateN"],
            {"input": window, "state": self._state, "sr": np.array([MODEL_RATE], dtype=np.int64)},
        )
        self._context = frame[-_CONTEXT:]
        return float(output[0, 0])
