"""A session's settings, as session.update sets them and session.updated shows them."""

from dataclasses import dataclass, field
from typing import Any

# OpenAI Realtime's PCM rate, both ways, until a session.update names another.
DEFAULT_RATE = 24_000
MIN_RATE = 8_000
MAX_RATE = 48_000


class RequestError(Exception):
    """A client event backtalk refuses: sent back as an `error` event."""

    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.param = param


@dataclass(frozen=True)
class TurnDetection:
    """Server VAD, with OpenAI's defaults. Automatic responses are always off."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    interrupt_response: bool = True


@dataclass(frozen=True)
class Settings:
    input_rate: int = DEFAULT_RATE
    output_rate: int = DEFAULT_RATE
    turn_detection: TurnDetection = field(default_factory=TurnDetection)


# Where a session.update can ask for automatic responses: the GA field and the
# beta one. conduit_push's read-back support looks for either name in `param`.
_TURN_DETECTION_PARAMS = (
    ("session.audio.input.turn_detection", ("audio", "input", "turn_detection")),
    ("session.turn_detection", ("turn_detection",)),
)


def updated(settings: Settings, session: Any) -> Settings:
    """`settings` with a session.update's `session` applied, all or nothing.

    Raises RequestError for an update backtalk can't honor. Asking for
    automatic responses (a turn_detection whose create_response isn't false,
    which is what leaving it out means) is refused with the code
    `realtime_live_unavailable`: backtalk has no model to answer with.
    Fields it has no use for (instructions, voice, tools, transcription) are
    left alone.
    """
    if not isinstance(session, dict):
        raise RequestError("invalid_type", "session must be an object", "session")
    for param, path in _TURN_DETECTION_PARAMS:
        block = _at(session, path)
        if isinstance(block, dict) and block.get("create_response", True) is not False:
            raise RequestError(
                "realtime_live_unavailable",
                "backtalk only transcribes and reads text back; set turn_detection.create_response to false",
                f"{param}.create_response",
            )
    audio = _object(session, "audio", "session.audio")
    audio_in = _object(audio, "input", "session.audio.input")
    audio_out = _object(audio, "output", "session.audio.output")
    turn_detection = settings.turn_detection
    if "turn_detection" in audio_in:
        turn_detection = _turn_detection(audio_in["turn_detection"])
    return Settings(
        input_rate=_rate(audio_in.get("format"), "session.audio.input.format", settings.input_rate),
        output_rate=_rate(audio_out.get("format"), "session.audio.output.format", settings.output_rate),
        turn_detection=turn_detection,
    )


def described(settings: Settings, session_id: str, model: str) -> dict[str, Any]:
    """The session object of session.created and session.updated."""
    detection = settings.turn_detection
    return {
        "type": "realtime",
        "object": "realtime.session",
        "id": session_id,
        "model": model,
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": settings.input_rate},
                "transcription": {"model": model},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": detection.threshold,
                    "prefix_padding_ms": detection.prefix_padding_ms,
                    "silence_duration_ms": detection.silence_duration_ms,
                    "create_response": False,
                    "interrupt_response": detection.interrupt_response,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": settings.output_rate}},
        },
    }


def _at(session: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = session
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _object(parent: dict[str, Any], key: str, param: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RequestError("invalid_type", f"{param} must be an object", param)
    return value


def _rate(audio_format: Any, param: str, current: int) -> int:
    if audio_format is None:
        return current
    if not isinstance(audio_format, dict):
        raise RequestError("invalid_type", f"{param} must be an object", param)
    kind = audio_format.get("type")
    if kind != "audio/pcm":
        raise RequestError("invalid_value", f"backtalk takes only audio/pcm, not {kind!r}", f"{param}.type")
    rate = audio_format.get("rate", DEFAULT_RATE)
    if not isinstance(rate, int) or isinstance(rate, bool) or not MIN_RATE <= rate <= MAX_RATE:
        raise RequestError(
            "invalid_value",
            f"rate must be a whole number of Hz from {MIN_RATE} to {MAX_RATE}, not {rate!r}",
            f"{param}.rate",
        )
    return rate


def _turn_detection(block: Any) -> TurnDetection:
    param = "session.audio.input.turn_detection"
    if not isinstance(block, dict):
        raise RequestError(
            "invalid_value", "backtalk always detects speech: turn_detection must be server_vad", param
        )
    kind = block.get("type", "server_vad")
    if kind != "server_vad":
        raise RequestError(
            "invalid_value", f"backtalk's turn detection is server_vad, not {kind!r}", f"{param}.type"
        )
    defaults = TurnDetection()
    threshold = block.get("threshold", defaults.threshold)
    if not isinstance(threshold, int | float) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise RequestError(
            "invalid_value", f"threshold must be from 0 to 1, not {threshold!r}", f"{param}.threshold"
        )
    padding = _milliseconds(block, "prefix_padding_ms", defaults.prefix_padding_ms, param, minimum=0)
    silence = _milliseconds(block, "silence_duration_ms", defaults.silence_duration_ms, param, minimum=1)
    interrupt = block.get("interrupt_response", defaults.interrupt_response)
    if not isinstance(interrupt, bool):
        raise RequestError(
            "invalid_type", "interrupt_response must be true or false", f"{param}.interrupt_response"
        )
    return TurnDetection(float(threshold), padding, silence, interrupt)


def _milliseconds(block: dict[str, Any], key: str, default: int, param: str, *, minimum: int) -> int:
    value = block.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RequestError(
            "invalid_value",
            f"{key} must be a whole number of at least {minimum}, not {value!r}",
            f"{param}.{key}",
        )
    return value
