"""Server VAD: where speech starts and stops in a session's input, frame by frame."""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from backtalk.audio import MODEL_RATE, Samples
from backtalk.settings import TurnDetection

# Silero VAD's frame at 16 kHz: 32 ms.
FRAME = 512
FRAME_MS = FRAME * 1000 // MODEL_RATE
# Speech starts after this many frames over the threshold in a row, so a
# click doesn't stop a reply.
START_FRAMES = 2
# A frame this far under the threshold is silence; between the two, the
# count of silent frames neither grows nor resets (Silero's hysteresis).
SILENCE_MARGIN = 0.15
# Speech that goes on this long is cut into another turn, so one session's
# buffer stays bounded however noisy the room.
MAX_SPEECH_FRAMES = 30_000 // FRAME_MS


@dataclass(frozen=True)
class SpeechStarted:
    start_ms: int


@dataclass(frozen=True)
class SpeechStopped:
    end_ms: int
    audio: Samples


class Segmenter:
    """Speech start and stop from per-frame speech probabilities, as OpenAI's
    server VAD reports them. Times count from the session's first frame."""

    def __init__(self, detection: TurnDetection) -> None:
        self._frames = 0
        self._over = 0
        self._silent = 0
        self._speech: list[Samples] | None = None
        self._recent: deque[Samples] = deque()
        self.configure(detection)

    def configure(self, detection: TurnDetection) -> None:
        self._detection = detection
        padding = math.ceil(detection.prefix_padding_ms / FRAME_MS)
        self._recent = deque(self._recent, maxlen=padding + START_FRAMES)
        self._silence_frames = max(1, math.ceil(detection.silence_duration_ms / FRAME_MS))

    @property
    def speaking(self) -> bool:
        return self._speech is not None

    def feed(self, frame: Samples, probability: float) -> SpeechStarted | SpeechStopped | None:
        self._frames += 1
        threshold = self._detection.threshold
        if self._speech is None:
            self._recent.append(frame)
            self._over = self._over + 1 if probability >= threshold else 0
            if self._over < START_FRAMES:
                return None
            self._speech = list(self._recent)
            self._recent.clear()
            self._over = 0
            self._silent = 0
            return SpeechStarted((self._frames - len(self._speech)) * FRAME_MS)
        self._speech.append(frame)
        if probability >= threshold:
            self._silent = 0
        elif probability < threshold - SILENCE_MARGIN:
            self._silent += 1
        if self._silent >= self._silence_frames or len(self._speech) >= MAX_SPEECH_FRAMES:
            return self.cut()
        return None

    def cut(self) -> SpeechStopped | None:
        """Ends the speech in progress here; None when there's none."""
        if self._speech is None:
            return None
        speech, self._speech = self._speech, None
        return SpeechStopped((self._frames - self._silent) * FRAME_MS, np.concatenate(speech))

    def clear(self) -> None:
        self._speech = None
        self._recent.clear()
        self._over = 0
