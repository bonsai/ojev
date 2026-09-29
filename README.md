# ojev

OpenJev-style semantic decision scoring for ordinary CPU machines.

This repository is the CPU-first edition of the OpenJev/SemIf experiment:
**AMD CPU + 16GB RAM, no GPU required. Slow inference is acceptable.**

The core experiment is intentionally small:

1. Give a model evidence/state.
2. Give a criterion/question.
3. Give 2–16 declared options.
4. Read the logits of the answer token for each option.
5. Softmax only across the declared options.

No answer generation is required for the direct scorer.

## Target machine

- AMD CPU
- 16GB RAM
- CPU-only
- Linux / WSL recommended
- Quantized GGUF model
- Small models first (Qwen3 0.6B is the default example)

The project does **not** require CUDA, NVIDIA, or a 3090.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
CMAKE_ARGS="-DGGML_NATIVE=ON" pip install llama-cpp-python
pip install -e .
```

On Windows, WSL is the simplest CPU build environment.

## Model

Use a local GGUF file. A Q4_K_M quantization of a small 0.6B–2B instruct model is the intended starting point.

Example layout:

```text
models/
  qwen3-0.6b-q4_k_m.gguf
```

Do not commit model weights to this repository.

## Run

```bash
python -m ojev \
  --model models/qwen3-0.6b-q4_k_m.gguf \
  --input examples/decisions.jsonl \
  --output results.jsonl
```

Optional CPU controls:

```bash
python -m ojev \
  --model models/qwen3-0.6b-q4_k_m.gguf \
  --threads 4 \
  --ctx-size 4096 \
  --input examples/decisions.jsonl \
  --output results.jsonl
```

## Input contract

Each JSONL row has:

```json
{
  "id": "example-1",
  "state": "Evidence available to the model.",
  "question": "Which option best satisfies the criterion?",
  "options": [
    {"id": "a", "description": "First option."},
    {"id": "b", "description": "Second option."}
  ]
}
```

## Output

Each row contains the conditional distribution over the declared options, plus timing and model metadata.

The probabilities are **option-relative model scores**, not calibrated real-world confidence.

## Research boundary

ojev is an independent experiment. It is not affiliated with Jev or TypeSafe.

The CPU implementation is deliberately separated from the original browser/WebGPU path so that CPU results can be reproduced on modest hardware.

```
decision
  state + criterion + options
          |
          v
      GGUF / CPU
          |
          v
   answer-token logits
          |
          v
 softmax(declared options)
          |
          v
       JSONL
```

## License

MIT.
