from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = "Estimate a probability distribution over the listed options."


def validate(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(row["options"], list) or not 2 <= len(row["options"]) <= 16:
        raise ValueError("options must contain 2-16 entries")
    ids = [o.get("id") for o in row["options"] if isinstance(o, dict)]
    if len(ids) != len(row["options"]) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("every option needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("option ids must be unique")


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def lmstudio_score(base_url: str, model: str, row: dict, temperature: float = 0.0) -> dict:
    validate(row)
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                f"{SYSTEM} Return JSON only. "
                'Schema: {"probabilities": {"option_id": number}}. '
                "Numbers must be nonnegative and sum to 1."
            )},
            {"role": "user", "content": json.dumps({
                "evidence": row["state"],
                "criterion": row["question"],
                "options": row["options"],
            }, ensure_ascii=False)},
        ],
        "temperature": temperature,
        "max_tokens": 256,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ojev_distribution",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "probabilities": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        }
                    },
                    "required": ["probabilities"],
                    "additionalProperties": False,
                },
            },
        },
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    token = os.environ.get("LM_API_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=600) as response:
        data = json.load(response)

    parsed = json.loads(data["choices"][0]["message"]["content"])
    raw = parsed["probabilities"]
    values = [float(raw[o["id"]]) for o in row["options"]]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise RuntimeError("LM Studio returned invalid probabilities")
    total = sum(values)
    if total <= 0:
        raise RuntimeError("LM Studio returned an empty probability distribution")
    values = [v / total for v in values]
    return {
        "id": row["id"],
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": values,
        "backend": "lmstudio",
        "model": model,
        "base_url": base_url,
        "seconds": time.perf_counter() - started,
        "probability_status": (
            "generated self-reported option distribution; "
            "not direct logits and uncalibrated"
        ),
    }


def llamacpp_score(llm, row: dict) -> dict:
    validate(row)
    started = time.perf_counter()
    prompt = (
        "Choose exactly one listed option. Respond only with its uppercase letter.\n\n"
        f"EVIDENCE:\n{row['state']}\n\nCRITERION:\n{row['question']}\n\n"
        + "\n".join(f"{LETTERS[i]}. {o['description']}" for i, o in enumerate(row["options"]))
        + "\n\nANSWER:"
    )
    tokens = llm.tokenize(prompt.encode(), add_bos=True)
    llm.reset()
    llm.eval(tokens)
    logits = llm.scores[-1]
    ids = []
    for letter in LETTERS[:len(row["options"])]:
        encoded = llm.tokenize(letter.encode(), add_bos=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{letter!r} is not a single token")
        ids.append(encoded[0])
    selected = [float(logits[i]) for i in ids]
    return {
        "id": row["id"],
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "answer_tokens": ids,
        "backend": "llama.cpp",
        "gpu_layers": 0,
        "seconds": time.perf_counter() - started,
        "probability_status": "conditional option score; uncalibrated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-first OpenJev scorer")
    parser.add_argument("--backend", choices=("lmstudio", "llamacpp"), default="lmstudio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")

    llm = None
    if args.backend == "llamacpp":
        if not Path(args.model).is_file():
            parser.error(f"model not found: {args.model}")
        from llama_cpp import Llama
        kwargs = {
            "model_path": str(args.model), "n_ctx": args.ctx_size,
            "n_batch": args.n_batch, "n_gpu_layers": 0,
            "logits_all": True, "verbose": False,
        }
        if args.threads:
            kwargs["n_threads"] = args.threads
            kwargs["n_threads_batch"] = args.threads
        llm = Llama(**kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as dest:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            result = (
                lmstudio_score(args.base_url, args.model, row, args.temperature)
                if args.backend == "lmstudio"
                else llamacpp_score(llm, row)
            )
            dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            dest.flush()


if __name__ == "__main__":
    main()
