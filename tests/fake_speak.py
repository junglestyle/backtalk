"""A speak worker for tests, speaking the worker contract (`speak.mjs --serve`).

Each word is 0.1 s of a 440 Hz tone at 48 kHz. Text with "[fail]" is refused,
"[die]" makes the worker exit without answering, and "[slow]" takes a second.
"""

import base64
import json
import math
import sys
import time

RATE = 48_000
SECONDS_PER_WORD = 0.1


def main():
    for line in sys.stdin:
        request = json.loads(line)
        text = request["text"]
        if "[die]" in text:
            sys.exit(3)
        if "[slow]" in text:
            time.sleep(1)
        if "[fail]" in text:
            reply = {"id": request["id"], "error": "this text can't be said"}
        else:
            samples = int(len(text.split()) * SECONDS_PER_WORD * RATE)
            pcm = b"".join(
                int(8000 * math.sin(2 * math.pi * 440 * n / RATE)).to_bytes(2, "little", signed=True)
                for n in range(samples)
            )
            reply = {"id": request["id"], "rate": RATE, "pcm": base64.b64encode(pcm).decode()}
        print(json.dumps(reply), flush=True)


main()
