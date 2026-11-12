# A fast model for live calls

backtalk has no language model, and a live call on it doesn't need one: by default the conduit_push plugin runs a
Hermes turn for each thing the owner says. That's slow to start talking. Conduit's agent mode (Voice mode > Local)
puts a fast model in front of Hermes instead. It holds the conversation, answers small talk itself, and hands real
work to Hermes as jobs, then tells the owner what Hermes found. This is the reference setup for that model on the same
GPU host as backtalk, and how it was chosen.

backtalk never talks to it. The plugin calls the fast model itself, for each turn backtalk transcribes, and sends its
words back to backtalk to read aloud (the plugin's README, "Agent mode: a fast model in front of Hermes").

## What it needs

- **OpenAI chat completions with tools.** The plugin sends the call's tools (`start_job`, `list_jobs`, `cancel_job`
  and the rest) and streams the reply.
- **No reasoning.** The plugin sends a plain request and can't ask a model not to think, so a model that reasons
  must have it turned off on the server, or every reply waits while it thinks.
- **A first token in well under a second,** since the owner hears nothing until it comes.
- **The right call.** Small talk answered, house questions handed to Hermes as a job, and a job's result relayed
  without inventing anything. Saying "let me check" without starting a job is the worst failure: the owner waits for
  an answer that never comes.

## The reference setup

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s server from its CUDA image, `ghcr.io/ggml-org/llama.cpp`
(`server-cuda`), pinned by digest, serving Mistral's own
[Ministral 3 8B Instruct 2512 GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF) at
`Q4_K_M` (5.2 GB, Apache 2.0):

```sh
llama-server -m /models/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf \
  --n-gpu-layers all --ctx-size 16384 --parallel 1 \
  --jinja --reasoning-budget 0 --temp 0.05 \
  --api-key "$KEY" --host 0.0.0.0 --port 8080
```

- `--jinja` uses the chat template in the GGUF, which is what parses the model's tool calls.
- `--reasoning-budget 0` keeps any reasoning off on the server. Ministral's Instruct model doesn't reason, so it's a
  guard for a model swapped in later.
- `--temp 0.05` is the server's default, below the 0.1 Mistral recommends for production. A request's own
  temperature still wins.
- `--parallel 1` gives one call the whole 16k context. A second request waits for the first.
- It holds 7.2 GB of GPU memory, beside backtalk's 3.9 GB.

Hermes's profile points agent mode at it. Conduit's Local settings write this:

```yaml
voice:
  local_live:
    front_model:
      base_url: http://192.0.2.10:8766/v1
      model: ministral
      api_key: <the --api-key>
```

The plugin sends plain `http://` only to private, Tailscale and LAN addresses, since the key goes with each request.

A GPU worker sharing the host, such as a corral worker, sees the server as a GPU process (`llama-server`), so it
needs it on its ignore list like backtalk, or it stops taking GPU work.

## How it was chosen

[`tools/call_model_eval.py`](../tools/call_model_eval.py) plays ten moments of a call to a server and judges each:
three it should answer itself (a greeting, a joke, arithmetic), four house questions it should hand to Hermes,
"what are you working on" and "cancel that" mid-call, and relaying a job's result with its numbers intact. A
`list_jobs` call gets its result and the model goes on, as on a real call. To rerun it against a server:

```sh
CALL_MODEL_KEY=<key> uv run python tools/call_model_eval.py --url http://192.0.2.10:8766/v1 --model ministral --runs 3
```

On an RTX 5070 Ti, with models that fit in about 8 GB of GPU memory beside backtalk, at `Q4_K_M` unless named
(Nemotron at `UD-Q4_K_XL`):

| Model | Passed | First token, median | GPU memory | Where it fell short |
| --- | --- | --- | --- | --- |
| Ministral 3 8B Instruct 2512 | 27/30 | 0.03 s | 7.2 GB, 16k context | Relays in rounded percentages, never invents |
| Nemotron 3.5 Bolt E4B | 24/30 | 0.07 s | 4.0 GB, 8k context | Routes perfectly; no arithmetic; drops "0 errors" |
| Qwen3.5 9B | 8/10 | 0.37 s* | 5.6 GB | Checked the job list instead of starting a job; turned TiB into percentages |
| Qwen3.5 4B | 7/10 | 0.39 s* | 3.1 GB | Sent a joke to Hermes, refused arithmetic, relay dropped every number |
| LFM3 1.7B (`Q8_0`) | 6–7/10 | 0.04 s | 2.2 GB | Only exact relay; says it'll check, then starts no job |
| LFM2.5 8B A1B (`Q5_K_M`) | 7/10 | 0.74 s* | 6.1 GB | Reasons anyway; claimed a restart without a job |

Rows marked * ran on Ollama, whose overhead the llama.cpp rows don't have, and the rows out of 10 ran once. Two
models that fit were left out: Gemma 4 12B's weights alone fill the 8 GB, and the Gemma 4 E2B and E4B models reason
for about a second before replying.

Ten moments played three times is a small sample, and the test's instructions aren't Conduit's. Real calls are the
real test, and swapping the model is a change of file and flags.
