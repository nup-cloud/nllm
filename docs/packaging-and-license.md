# Packaging and License

## What is in this OSS repository

Everything in this repository is released under **Apache 2.0**:

- **Runtime core** -- sanitizer, whitelist, engine protocol (`src/nllm/core/`)
- **Device modules** -- drone, camera, safety policies (`src/nllm/device/`)
- **Sensor pipeline** -- schema validation, range checks, dedup (`src/nllm/sensor/`)
- **Command layer** -- prompt builder, LLM output parser (`src/nllm/command/`)
- **Planning** -- task planner, executor, agent memory (`src/nllm/planning/`)
- **Alert engine** -- rule-based alerting (`src/nllm/alert/`)
- **Adapters** -- Ollama, llama-cpp, MQTT, ROS2 (`src/nllm/adapters/`)
- **Skills** -- domain skill files (`skills/`)
- **Simulator** -- device simulation for testing (`src/nllm/simulator/`)
- **Evaluation** -- accuracy benchmarking framework (`src/nllm/eval/`)
- **Training recipes** -- QLoRA fine-tuning scripts and sample data (`training/`)

This repo contains no non-OSS license restrictions.

## Upstream Model Licenses

nllm does not ship model weights. Base models you download carry their own
licenses. See [MODEL_ORIGIN.md](../MODEL_ORIGIN.md) for details on tested
bases and their license obligations.

## What May Be Offered Separately in Future

The following are not part of this repository and may be offered as separate
commercial products:

- **Agent Pro** -- managed multi-agent orchestration service
- **Hardware Connectors** -- certified plugins for DJI SDK, ONVIF, proprietary protocols
- **GCP/AWS Ops** -- cloud deployment, monitoring, and scaling recipes
- **Enterprise Security** -- SSO, RBAC policy management, audit export
- **Private Adapters** -- hosted LoRA training and adapter management
