# Call Me Maybe — function calling with constrained decoding

[![CI](https://github.com/imyassmine/call-me-maybe/actions/workflows/ci.yml/badge.svg)](https://github.com/imyassmine/call-me-maybe/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-261230.svg)](https://docs.astral.sh/uv/)

**Turn a natural-language request into a schema-valid function call — on a 0.6 B model, on CPU, with no fine-tuning.**
Invalid function names and malformed arguments are impossible by construction: instead of hoping the
model emits valid JSON, generation is constrained token by token with a finite-state machine, and
Python assembles the final call.

| Prompt | Generated call |
| --- | --- |
| `What is the sum of 2 and 3?` | `fn_add_numbers(a=2, b=3)` |
| `Greet shrek` | `fn_greet(name="Shrek")` |
| `Reverse the string 'hello'` | `fn_reverse_string(s="hello")` |

## Highlights

- **Constrained decoding from scratch**: at every step, only tokens that keep the output valid are
  eligible for greedy selection. The model decides *what* to say; the FSM enforces *how*.
- **Two-phase generation**: function selection is prefix-constrained against the declared function
  names, then each argument is generated under its JSON type (`number`, `string`, `boolean`).
- **The LLM does only what needs language understanding**: choosing the function and extracting
  values. JSON punctuation, parameter order and Python type conversion are deterministic code.
- **Robust on a small model**: early-stop heuristic for unbounded numeric values, one explicit error
  type per failure phase, and schema re-validation of every result.
- **Quality gates**: 41 offline unit tests (no GPU, no model download) and an executable schema
  checker. GitHub Actions runs flake8, mypy and pytest on every push, plus a manual end-to-end model
  smoke test.

## Results

On the 11 provided prompts with Qwen3-0.6B (CPU only):

- **11/11** calls are schema-compliant — a property of the decoder, not a lucky run.
- **9/11** are also semantically correct; the two misses are regex substitutions approximated by the
  0.6 B model.
- **< 30 s per prompt** on a 4-core CPU with the model loaded (~5 min for the full set).

## Quick start

Python 3.10+ and [uv](https://docs.astral.sh/uv/). The first run downloads Qwen3-0.6B (~1.5 GB) and
caches it.

```bash
git clone https://github.com/imyassmine/call-me-maybe.git && cd call-me-maybe
make install   # uv sync
make run       # run the model on the provided fixtures
make test      # offline unit tests
make lint      # flake8 + mypy
```

Results are written to `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  }
]
```

Custom inputs and outputs: `make run ARGS="--input my_prompts.json --output my_results.json"`.

## Architecture

```
src/generate.py           two-phase generation (function + typed arguments)
src/value_conversion.py   raw value → typed Python value
src/state_machine.py      constrained-decoding FSM (valid continuations per state)
src/token_selection.py    highest-valid-logit selection + early stopping
src/prompts.py            prompt templates
src/json_file_parsing.py  Pydantic models for functions and fixtures
scripts/verify_results.py schema validation of generated calls
llm_sdk/                  thin Transformers inference wrapper (vendored)
```

**Stack:** Python 3.10 · PyTorch · Transformers · Qwen3-0.6B · Pydantic · uv · pytest · mypy ·
flake8 · GitHub Actions

**Next steps:** larger model for semantic accuracy, batched prompts with KV-cache reuse,
enum/array/nested-object parameter types.

---
*Built at [42 Paris](https://42.fr) by [Yassmine Abdallah (@imyassmine)](https://github.com/imyassmine).*
