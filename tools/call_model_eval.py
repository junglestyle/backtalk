"""Checks a fast model for Conduit's agent mode against the job it does on a call (docs/call-model.md).

It plays ten moments of a call to an OpenAI-compatible chat-completions server, each with a right answer: small talk
and arithmetic it should answer itself, house questions it should hand to Hermes with start_job, "what are you working
on" and "cancel that" mid-call, and a job's result it should relay with its numbers intact. A list_jobs call gets its
result back and the model goes on, as on a real call. It prints each moment's outcome to stderr and a JSON summary
per run to stdout.

    CALL_MODEL_KEY=... uv run python tools/call_model_eval.py --url http://host:8766/v1 --model ministral --runs 3
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SYSTEM = """You are the owner's voice assistant on a live phone call, the quick front of Hermes, their home agent.
Keep replies to one or two short spoken sentences, no lists or markdown.
You know only the conversation. Anything about the house (the TrueNAS NAS with the pools "fast" and "storage", its
apps hermes, corral, hearsay and tailscale, the GPU host, backups, replication, files, logs) needs Hermes:
call start_job with the owner's request restated precisely in their own terms, then say briefly that you're checking.
Never guess facts about the house. Use list_jobs to say what's running and cancel_job when the owner calls one off.
When a job's result arrives, tell the owner what Hermes found, keeping every number exactly as given."""


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    parameters = {"type": "object", "properties": properties, "required": list(properties)}
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


TOOLS = [
    _tool(
        "start_job",
        "Hand a task to Hermes, the full agent. Returns the job id.",
        {"task": {"type": "string"}},
    ),
    _tool("list_jobs", "List Hermes's running jobs.", {}),
    _tool("cancel_job", "Cancel a running Hermes job.", {"job_id": {"type": "string"}}),
]

# A call already under way: a question handed to Hermes as job-42, still running.
DURING_JOB: list[dict[str, Any]] = [
    {"role": "user", "content": "how are the pools doing"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "start_job",
                    "arguments": '{"task": "Check the health and space of the pools"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": '{"job_id": "job-42", "status": "running"}'},
    {"role": "assistant", "content": "I'm checking on the pools now."},
]
RUNNING = [{"job_id": "job-42", "task": "Check the health and space of the pools"}]
RESULT = (
    "Job job-42 finished. Hermes reports: pool fast is ONLINE, 1.82 TiB used of 3.62 TiB (50%), last scrub "
    "on November 9 found 0 errors. Pool storage is ONLINE with 9.1 TiB used of 21.8 TiB."
)


@dataclass(frozen=True)
class Reply:
    seconds_to_first_token: float
    calls: list[str]
    text: str  # what it said, then each call's arguments


def _plain(text: str) -> str:
    return text.lower().replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-")


def _answers_itself(reply: Reply) -> bool:
    return not reply.calls


def _hands_off(*words: str) -> Callable[[Reply], bool]:
    def check(reply: Reply) -> bool:
        text = _plain(reply.text)
        return reply.calls[:1] == ["start_job"] and all(word in text for word in words)

    return check


@dataclass(frozen=True)
class Case:
    said: str
    history: list[dict[str, Any]]
    passes: Callable[[Reply], bool]


CASES = [
    Case("hey what's up", [], _answers_itself),
    Case("tell me a joke", [], _answers_itself),
    Case(
        "what's seventeen times three",
        [],
        lambda r: not r.calls and any(n in _plain(r.text) for n in ("51", "fifty-one", "fifty one")),
    ),
    Case("is the fast pool healthy", [], _hands_off("fast")),
    Case(
        "how much space is left on the fast pool",
        [],
        lambda r: _hands_off("fast")(r) and "azure" not in _plain(r.text),
    ),
    Case("did last night's replication to storage finish", [], _hands_off("replicat")),
    Case("restart the hearsay container", [], _hands_off("hearsay")),
    Case(
        "what are you working on right now",
        DURING_JOB,
        lambda r: r.calls[:1] == ["list_jobs"] or (not r.calls and "pool" in _plain(r.text)),
    ),
    Case(
        "actually never mind, cancel that",
        DURING_JOB,
        lambda r: r.calls[:1] == ["cancel_job"] and "job-42" in r.text,
    ),
    Case(
        RESULT,
        DURING_JOB,
        lambda r: (
            not r.calls
            and "1.82" in r.text
            and "3.62" in r.text
            and any(e in _plain(r.text) for e in ("0 errors", "no errors", "zero errors"))
        ),
    ),
]


@dataclass(frozen=True)
class Server:
    url: str  # ends in /v1
    model: str
    key: str | None
    temperature: float | None


def _ask(server: Server, messages: list[dict[str, Any]]) -> Reply:
    body: dict[str, Any] = {
        "model": server.model,
        "stream": True,
        "tools": TOOLS,
        "messages": [{"role": "system", "content": SYSTEM}, *messages],
    }
    if server.temperature is not None:
        body["temperature"] = server.temperature
    headers = {"Content-Type": "application/json"}
    if server.key:
        headers["Authorization"] = f"Bearer {server.key}"
    request = urllib.request.Request(  # noqa: S310 - main() takes only an http(s) --url
        f"{server.url}/chat/completions", json.dumps(body).encode(), headers, method="POST"
    )
    start = time.monotonic()
    first: float | None = None
    content = ""
    calls: dict[int, dict[str, str]] = {}
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310 - as above
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line.removeprefix("data: ")).get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            if first is None and (delta.get("content") or delta.get("tool_calls")):
                first = time.monotonic() - start
            content += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    elapsed = time.monotonic() - start
    text = " ".join([content, *(c["arguments"] for c in calls.values())]).strip()
    return Reply(first if first is not None else elapsed, [c["name"] for c in calls.values()], text)


def _play(server: Server, case: Case) -> Reply:
    asked = [*case.history, {"role": "user", "content": case.said}]
    reply = _ask(server, asked)
    if reply.calls == ["list_jobs"] and not case.passes(reply):
        # A real call returns the tool's result and lets the model go on.
        listed = {"id": "l1", "type": "function", "function": {"name": "list_jobs", "arguments": "{}"}}
        jobs = json.dumps(RUNNING if case.history else [])
        more = [
            *asked,
            {"role": "assistant", "content": "", "tool_calls": [listed]},
            {"role": "tool", "tool_call_id": "l1", "content": jobs},
        ]
        after = _ask(server, more)
        reply = Reply(reply.seconds_to_first_token, after.calls, after.text)
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="the server's OpenAI-compatible base URL, ending in /v1")
    parser.add_argument("--model", required=True, help="the model name the server answers to")
    parser.add_argument("--runs", type=int, default=3, help="how many times to play the call (default 3)")
    parser.add_argument(
        "--temperature", type=float, help="sent with each request; by default the server's own is used"
    )
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        raise SystemExit(f"--url must be an http or https address, not {args.url!r}")
    server = Server(args.url.rstrip("/"), args.model, os.environ.get("CALL_MODEL_KEY"), args.temperature)

    _ask(server, [{"role": "user", "content": "hello"}])  # so loading the model isn't timed
    for run in range(1, args.runs + 1):
        passed = 0
        firsts = []
        for case in CASES:
            reply = _play(server, case)
            ok = case.passes(reply)
            passed += ok
            firsts.append(reply.seconds_to_first_token)
            outcome = "PASS" if ok else "FAIL"
            print(
                f"{outcome} {reply.seconds_to_first_token:.2f}s {case.said[:48]!r} calls={reply.calls} {reply.text[:100]!r}",
                file=sys.stderr,
            )
        summary = {
            "model": server.model,
            "run": run,
            "passed": passed,
            "of": len(CASES),
            "first_token_median_s": round(statistics.median(firsts), 3),
            "first_token_worst_s": round(max(firsts), 3),
        }
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
