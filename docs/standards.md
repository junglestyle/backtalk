# Coding standards

backtalk follows the [org standards](org-standards.md) and the standards for its topics
([containers](org-standards-containers.md), [python](org-standards-python.md)), unchanged copies of
[junglestyle's](https://github.com/junglestyle/.github/tree/main/code-standards). `.org-standards` lists the topics
it copies beyond the general one. This file adds backtalk's own rules and examples. Where it differs from an org
rule, it says so, and it wins.

## Structure

- The layers, and what each mustn't know:
  - `server.py`: the socket, the key and one session per connection. It knows nothing of the protocol's events.
  - `session.py`: the protocol, one connection's events in and out. It reaches the models and the speak worker only
    through the Protocols it defines (`Transcriber`, `Detector`, `Speaker`), and knows nothing of onnxruntime,
    Hugging Face or Node.
  - `settings.py` (what `session.update` may set), `vad.py` (where speech starts and stops) and `audio.py` (wire
    audio and rates) are plain functions and data the session uses.
  - `models.py` and `speak.py`: the real models and worker behind those Protocols.
- The protocol follows what conduit_push's read-back support and Conduit's speech server clients send and check.
  An event, field or error code they rely on is interface: change it only against their source.

## Concurrency

- Overrides the Python standard's "synchronous by default": the sockets run on asyncio, since `websockets` is an
  async library and each connection waits on its client, its turns and its readings at once. Blocking work stays
  synchronous and off the loop: a transcription runs in a thread, and Silero's per-frame scoring is small enough to
  run on it.

## Dependencies

- onnxruntime comes from `onnxruntime-gpu` on Linux, where the image runs it on CUDA, and from `onnxruntime`
  elsewhere, since the GPU build has no Mac wheels. Both install the same module.

## Running it

- Overrides the container standard's base image: the image starts from NVIDIA's CUDA runtime with cuDNN, pinned by
  digest, because onnxruntime-gpu loads CUDA's and cuDNN's libraries from the system and no Debian slim image has
  them. uv installs the Python `.python-version` names into the image.

## Testing

- The behaviors worth a test, all over a real WebSocket against a fake model and a fake speak worker
  (`tests/fake_speak.py`, which speaks the worker's line protocol): a turn's speech events and transcript, a
  read-back response's events and audio at the session's rate, cancel and barge-in, the refusal of automatic
  responses, a speak worker that dies, the bearer key, and two sessions at once.
- The real models are checked by hand, as the README's "Locally" runs them; `make check` never downloads them.
