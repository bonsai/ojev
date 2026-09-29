from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


SYSTEM = (
    "Apply the criterion to the evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter."
)
LETTERS = "ABCDEFGHIJKLMNOP"


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


def prompt(row: dict) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {o['description']}" for i, o in enumerate(row["options"])
    )
    return (
        f"{SYSTEM}\n\n"
        f"EVIDENCE:\n{row['state']}\n\n"
        f"CRITERION:\n{row['question']}\n\n"
        f"OPTIONS:\n{options}\n\n"
        "ANSWER:"
    )


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def score(llm, row: dict) -> dict:
    validate(row)
    text = prompt(row)
    started = time.perf_counter()

    # llama.cpp exposes the final-position vocabulary logits after eval().
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(tokens)
    logits = llm.scores[-1]

    token_ids = []
    for letter in LETTERS[:len(row["options"])]:
        encoded = llm.tokenize(letter.encode("utf-8"), add_bos=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"answer slot {letter!r} is not a single GGUF token; "
                "use a model/tokenizer with single-token answer slots"
            )
        token_ids.append(encoded[0])

    selected = [float(logits[token_id]) for token_id in token_ids]
    probabilities = softmax(selected)

    return {
        "id": row["id"],
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": probabilities,
        "option_logits": selected,
        "answer_tokens": token_ids,
        "backend": "llama.cpp",
        "gpu_layers": 0,
        "input_tokens": len(tokens),
        "prompt": text,
        "seconds": time.perf_counter() - started,
        "probability_status": "conditional option score; uncalibrated decision confidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-first OpenJev scorer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=256)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    from llama_cpp import Llama

    kwargs = {
        "model_path": str(args.model),
        "n_ctx": args.ctx_size,
        "n_batch": args.n_batch,
        "n_gpu_layers": 0,
        "logits_all": True,
        "verbose": False,
    }
    if args.threads is not None:
        kwargs["n_threads"] = args.threads
        kwargs["n_threads_batch"] = args.threads

    llm = Llama(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as dest:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            result = score(llm, row)
            dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            dest.flush()


if __name__ == "__main__":
    main()
