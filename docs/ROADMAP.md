# Roadmap

The order backtalk gets built in. Each slice is usable on its own, names what a user would notice and the
[spec](SPEC.md) requirements it delivers, and has a done-when someone can demonstrate. A requirement not yet in a
slice stays in the spec and waits for one here.

## 1. A read-back speech server for hermes' calls

Status: planned

A Local Live call or Classic voice in Conduit, pointed at backtalk through conduit_push, is heard in the house and
answered by hermes in Chimera's voice: the user speaks, backtalk transcribes the turn on the GPU, hermes answers,
and each sentence of the reply is read aloud as it comes. Talking over the reply or tapping Interrupt stops it.

Delivers: CONN-1, CONN-2, CONN-3, SESSION-1, SESSION-2, SESSION-3, LISTEN-1, LISTEN-2, LISTEN-3, LISTEN-4,
READ-1, READ-2, READ-3, READ-4, READ-5, READ-6, SPEAK-1, SPEAK-2, RUN-1, RUN-2, RUN-3.

Done when: `make check` passes, driving the server over a real WebSocket with a fake model and a fake speak
worker; the published image runs on the GPU host and transcribes on the GPU; conduit_push's speech server test
connects; and a Local Live call from Conduit carries a spoken turn to hermes and plays the reply in Chimera's
voice, stops when talked over, and goes on to the next turn.

## 2. Words as they're spoken

Status: planned

The user sees their words appear while they're still talking, as with the phone's own recognition.

Delivers: LISTEN-5.

Done when: a turn of several seconds sends transcription deltas before `speech_stopped`, and its completed
transcript is no worse than without them.

## 3. Chimera starts talking sooner

Status: planned

A reply's first words play while the rest of its sentence is still being made, so a long sentence doesn't hold up
the start. It needs the speak worker to hand back audio in pieces.

Delivers: READ-7.

Done when: on the GPU host, the first audio of a 200-character sentence goes out in well under the time the
whole sentence takes to make.
