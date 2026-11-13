# Working on backtalk

A realtime speech server for the house: it hears and transcribes speech on the GPU and reads replies aloud in Chimera's voice, over the OpenAI Realtime protocol

Read before changing behavior:

- [README](README.md): what backtalk does and how to run it
- [Spec](docs/SPEC.md) and [roadmap](docs/ROADMAP.md): what it's meant to do, and the order it gets built in
- [Coding standards](docs/standards.md): how the code should read, on top of the org's
  [principles](https://github.com/lowmagic/.github/blob/main/PRINCIPLES.md) and standards (`docs/org-standards*.md`)

Docs live in `docs/` and change with behavior. Never commit hostnames, tokens, or deployment config (see
Privacy in the [org standards](docs/org-standards.md)).

Run `make check` before committing. It runs everything CI runs, except the org standards check:
`make standards` (needs the network).
