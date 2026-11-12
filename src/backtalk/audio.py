"""Audio on the wire (base64 PCM16, little-endian mono) and between rates."""

import base64
import binascii
from typing import Final

import numpy as np
import numpy.typing as npt
import soxr

# What the speech detector and the speech-to-text model hear.
MODEL_RATE: Final = 16_000

Samples = npt.NDArray[np.float32]
Pcm16 = npt.NDArray[np.int16]


def decode(audio: str) -> Pcm16:
    """Base64 PCM16 as samples; ValueError for anything else."""
    try:
        data = base64.b64decode(audio, validate=True)
    except binascii.Error as error:
        raise ValueError(f"audio isn't base64: {error}") from error
    if len(data) % 2:
        raise ValueError(f"audio is {len(data)} bytes, which isn't whole 16-bit samples")
    return np.frombuffer(data, dtype="<i2").astype(np.int16)


def encode(pcm: Pcm16) -> str:
    return base64.b64encode(pcm.astype("<i2").tobytes()).decode("ascii")


def resampled(pcm: Pcm16, rate: int, to_rate: int) -> Pcm16:
    if rate == to_rate:
        return pcm
    out: Pcm16 = soxr.resample(pcm, rate, to_rate)
    return out


class ToModelRate:
    """A session's input audio, as it streams in, at the models' rate."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._stream = soxr.ResampleStream(rate, MODEL_RATE, 1, dtype="float32")

    def __call__(self, pcm: Pcm16) -> Samples:
        samples = pcm.astype(np.float32) / 32768.0
        if self.rate == MODEL_RATE:
            return samples
        out: Samples = self._stream.resample_chunk(samples)
        return out
