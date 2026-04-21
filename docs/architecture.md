# Architecture

## Pipeline

```
Japanese natural-language input
  |
  v
[1] Sanitize        src/nllm/core/sanitizer.py
  |                 Reject prompt injection, strip control chars, enforce length limit.
  v
[2] Infer           src/nllm/adapters/ (ollama | llama_cpp)
  |                 Send cleaned prompt + skill context to the base model.
  v
[3] Parse           src/nllm/command/parser.py
  |                 Extract structured JSON command from model output.
  v
[4] Validate        src/nllm/core/whitelist.py, src/nllm/sensor/schema.py
  |                 Whitelist check (action + domain) and parameter range validation.
  v
[5] Plan            src/nllm/planning/plan.py
  |                 Build an execution plan; multi-step tasks become ordered steps.
  v
[6] Execute         src/nllm/planning/executor.py
                    Dispatch commands to device adapters (MQTT, ROS2, etc.).
                    Human-in-the-loop approval for dangerous actions.
                    All decisions logged to src/nllm/planning/memory.py.
```

## Module Map

```
src/nllm/
├── types.py        Domain primitives and Result monad
├── core/           Engine protocol, sanitizer, whitelist
├── command/        Prompt builder, LLM output parser
├── device/         Drone, camera, safety policy
├── sensor/         Schema validation, range check, dedup
├── planning/       Task plans, executor, agent memory
├── alert/          Rule engine, alert pipeline
└── adapters/       Ollama, llama-cpp, MQTT, ROS2
```

## Key Design Decisions

- **Pure functions first** -- sanitizer and validators are stateless.
- **Result monad** -- errors propagate as values, not exceptions.
- **Offline-only** -- no network calls during inference or execution.
- **Base-model agnostic** -- any GGUF model works via the adapter layer.
