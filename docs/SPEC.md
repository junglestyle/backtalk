# Specification

## Problem

The household talks to hermes, the Hermes agent on the NAS, by voice from Conduit on an iPhone or an Apple Watch,
and wants to hear it answer in Chimera's voice. Conduit's Local Live calls and its Classic voice reach a voice of
the user's own only through a speech server: the conduit_push plugin on the Hermes host relays OpenAI Realtime
events between Conduit and that server. Without one, a call's speech is recognized by Apple or a cloud provider
and its replies are spoken in an iPhone voice.

The speech servers conduit_push knows (Hugging Face speech-to-speech, say) bring a language model of their own,
which would answer in hermes' place, and none of them speaks in Chimera's voice. Chimera's voice exists as
chimera-voice-chat's speak build, which turns text into audio and doesn't listen.

backtalk is the house's speech server. It runs on the GPU host, hears and transcribes speech there, and reads text
aloud in Chimera's voice through the speak build. It has no language model: conduit_push supports exactly such a
read-back server, running each turn on Hermes and sending the reply back a sentence at a time to be read. Each
requirement has an ID the [roadmap](ROADMAP.md) cites. A deferred requirement stays here and gets a later slice
there.

## Connections

- **CONN-1: One realtime endpoint.** backtalk serves the OpenAI Realtime WebSocket protocol at `/v1/realtime` on
  port 8765. Any other path is answered with HTTP 404.
- **CONN-2: A bearer key.** When `BACKTALK_API_KEY` is set, a connection that doesn't send
  `Authorization: Bearer <key>` is refused with HTTP 401 before the WebSocket opens, which conduit_push reports
  as the server refusing its key.
- **CONN-3: One session per connection, many at once.** Each connection is its own session and opens with
  `session.created`, the first event conduit_push's connection test waits for. Sessions run side by side: Classic
  voice opens a listening socket and a speaking socket together.

## Sessions

- **SESSION-1: Settings.** `session.update` sets the input and output audio (`audio/pcm`, mono 16-bit, at any rate
  from 8000 to 48000 Hz; 24000 until set) and the server VAD (`threshold`, `prefix_padding_ms`,
  `silence_duration_ms`, `interrupt_response`, with OpenAI's defaults of 0.5, 300, 500 and true). backtalk answers
  with `session.updated`, which shows the whole session. Fields it has no use for, such as `instructions`,
  `voice` and `tools`, are ignored. An update it can't honor is answered with an `error` that names the field in
  `param`, and changes nothing.
- **SESSION-2: Automatic responses are refused.** A `session.update` whose turn detection asks for automatic
  responses (`create_response` true, or left out, under `audio.input.turn_detection` or the beta
  `turn_detection`) is answered with an `error` whose code is `realtime_live_unavailable` and whose `param` is
  that field, and changes nothing. That is how conduit_push recognizes a read-back server: it then sends the same
  update with `create_response` and `interrupt_response` off and runs the call's turns itself.
- **SESSION-3: Errors.** A client event backtalk can't take (not JSON, an unknown `type`, a bad value) is answered
  with an `error` event of type `invalid_request_error`, carrying a code, a message, the `param` at fault, and
  the client's `event_id`.

## Listening

- **LISTEN-1: Speech detection on the server.** Input audio streams in as `input_audio_buffer.append`. backtalk
  finds where speech starts and stops with Silero VAD and reports each turn as
  `input_audio_buffer.speech_started` (`audio_start_ms`, `item_id`), then `input_audio_buffer.speech_stopped`
  (`audio_end_ms`) and `input_audio_buffer.committed` (`item_id`, `previous_item_id`). Times count from the
  session's first audio. Speech starts after 64 ms over the threshold and takes `prefix_padding_ms` of the audio
  before it; it stops after `silence_duration_ms` under the threshold less 0.15. A turn that runs 30 s is cut
  there and a new one starts.
- **LISTEN-2: Transcription.** Each committed turn is transcribed and reported as
  `conversation.item.input_audio_transcription.completed` with its `item_id` and `transcript`, in the order the
  turns were committed. A turn that can't be transcribed is reported as
  `conversation.item.input_audio_transcription.failed`. Turns from every session share one model, loaded once.
- **LISTEN-3: Commit and clear.** `input_audio_buffer.commit` ends the speech in progress as a turn. With none in
  progress it's answered with an `error` coded `input_audio_buffer_commit_empty`, which Conduit's Classic voice
  expects after its trailing silence. `input_audio_buffer.clear` drops the speech in progress and is answered
  with `input_audio_buffer.cleared`.
- **LISTEN-4: A pinned model on the GPU.** Speech is transcribed by NVIDIA's Parakeet TDT 0.6B v3 on the GPU,
  through onnxruntime's CUDA build; an image that can't reach the GPU stops at start rather than transcribe on the
  CPU. The model and Silero VAD are pinned to a Hugging Face commit and checked by sha256. They download into
  `BACKTALK_CACHE` on the first start and load from there after.
- **LISTEN-5: Words while speaking.** While a turn is still being spoken, its words so far arrive as
  `conversation.item.input_audio_transcription.delta`, which Conduit shows as they come.

## Reading back

- **READ-1: Text read aloud in Chimera's voice.** A `response.create` reads text aloud through the speak worker.
  The text is the user messages in the response's `input` (conduit_push's read-back sentence, sent with
  `conversation: "none"`), or, with no `input`, the newest user message a `conversation.item.create` added (Classic
  voice's speak socket). A response with no text to read is refused with an `error` coded `missing_text`.
- **READ-2: A response's events.** A read-back response sends `response.created`, then
  `response.output_audio_transcript.delta` with the whole text, then the audio as
  `response.output_audio.delta` events (base64 16-bit PCM at the session's output rate, 200 ms each), then
  `response.output_audio.done`, `response.output_audio_transcript.done` and `response.done` with status
  `completed`. The transcript is exactly the text given, which both conduit_push and Conduit check against what
  they sent.
- **READ-3: One response at a time.** A `response.create` while a response is in progress is refused with an
  `error` coded `conversation_already_has_active_response`; conduit_push sends the sentence again after that
  response's end.
- **READ-4: Cancel.** `response.cancel` ends the response in progress at once with `response.done` status
  `cancelled` (reason `client_cancelled`), and nothing more of it is sent. With no response in progress, it's
  answered with an `error` coded `response_cancel_not_active`, which conduit_push keeps to itself.
- **READ-5: Barge-in.** When speech starts while a response is in progress and the session's
  `interrupt_response` is on, the response ends as cancelled with reason `turn_detected`. conduit_push turns
  `interrupt_response` off and stops a reply itself with `response.cancel`, after the `speech_started` it sees.
- **READ-6: A failed reading.** A response the speak worker can't say ends with `response.done` status `failed`
  and an error coded `speak_failed`, so Conduit says the reply on the phone instead.
- **READ-7: Audio before the whole sentence is ready.** A response's first audio goes out before the speak
  worker has said the rest of its text.

## Speak worker

- **SPEAK-1: One long-lived worker.** backtalk runs `node $BACKTALK_SPEAK $BACKTALK_SPEAK_FLAGS --serve` as one
  child process, which loads Chimera's voice once and says one text at a time, over the line protocol of
  chimera-voice-chat's speak build. The flags are the voice's settings, such as its speed, which the build checks.
  It starts with backtalk, which says a word through it before taking connections and stops if it can't.
- **SPEAK-2: Restarted when it fails.** A worker that exits, gives no answer in 60 s, or answers something else
  than the request is stopped and started again for the next request. The request it failed ends as in READ-6.

## Running

- **RUN-1: A container image.** CI publishes `ghcr.io/lowmagic/backtalk:<full commit SHA>` from main. It runs
  on the GPU host with the GPU passed in, the speak build mounted read-only, and `BACKTALK_CACHE` on a persistent
  volume. The README says how.
- **RUN-2: Logs without words.** backtalk writes plain lines to stderr: models loaded, sessions opened and
  closed, and how long each transcription and reading took. It never logs what anyone said or what was read.
- **RUN-3: A clean stop.** SIGTERM stops new connections, ends the speak worker and exits 0.

## Unknowns

What the spec assumes without having checked:

- That onnxruntime-gpu's CUDA 13 build runs Parakeet on the GPU host's RTX 5070 Ti (Blackwell), with a driver
  for CUDA 13. backtalk has run only on a Mac's CPU, where a 3 s utterance was transcribed in 0.2 to 0.6 s.
- That `nvidia-smi` names backtalk's process `backtalk-server`, the name the image gives it so a GPU worker on
  the same host can ignore it.
- That Conduit's echo cancellation keeps Chimera's voice out of the microphone during a reply. If it doesn't,
  backtalk hears the reply as speech and conduit_push stops it.
- That the speak worker reads a sentence fast enough on the GPU host's CPU. On a Mac it took 0.9 s for 56
  characters, all of which comes before the sentence's first audio until READ-7.

## Open questions

Raised here, not decided:

- **Holding the GPU.** backtalk keeps Parakeet in GPU memory for as long as it runs, so a corral GPU worker on the
  same host either ignores it or stops taking GPU work. Unloading the model when no session is open would free
  the GPU between calls at the cost of a slow first turn.
- **Faster transcription.** Parakeet runs in fp32. fp16 or TensorRT might cut a turn's latency, at the cost of a
  second pinned export.
