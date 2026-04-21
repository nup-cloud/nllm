# nllm

Convert Japanese natural language into validated, structured IoT device commands -- fully offline, on any GGUF base model.

## Example

```
Input:  "ドローンを高度10mまで上昇させてください"
Output: {"domain": "drone", "action": "ASCEND", "params": {"altitude_m": 10}}
```

The input passes through a six-layer safety pipeline before any device receives a command: prompt-injection detection, command whitelisting, parameter range validation, device safety policies, human-in-the-loop approval for dangerous actions, and full audit logging. See [docs/architecture.md](docs/architecture.md) for the detailed flow.

## Architecture

```
src/nllm/
├── core/       Engine protocol, input sanitizer, command whitelist
├── command/    Prompt builder, LLM output parser
├── device/     Drone, camera, safety policy
├── sensor/     Schema validation, range check, dedup
├── planning/   Task planner, executor, agent memory
├── alert/      Rule engine, alert pipeline
└── adapters/   Ollama, llama-cpp, MQTT, ROS2
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Base Models

nllm is model-agnostic and works with any GGUF-compatible LLM.
See [docs/SUPPORTED_BASES.md](docs/SUPPORTED_BASES.md) for tested models
and instructions on adding your own.

## OSS vs Pro

This repository is the complete open-source runtime under Apache 2.0.
Hardware connectors, managed orchestration, and enterprise operations
may be offered separately in the future.
See [docs/packaging-and-license.md](docs/packaging-and-license.md) for details.

## Links

- [ROADMAP.md](ROADMAP.md) -- release plan
- [MODEL_ORIGIN.md](MODEL_ORIGIN.md) -- upstream model provenance and licenses
- [CONTRIBUTING.md](CONTRIBUTING.md) -- how to contribute
- [docs/security-architecture.md](docs/security-architecture.md) -- threat model and defence layers

## License

Apache 2.0. Model weights follow their upstream license -- see [MODEL_ORIGIN.md](MODEL_ORIGIN.md).
