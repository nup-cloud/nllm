# Model Origin and Provenance

This document records the upstream base models used or supported by nllm.

nllm itself is a **control runtime**, not a model. It works with any
GGUF-compatible base model. The models listed below have been tested
and are known to work well for Japanese IoT command generation.

## Tested Base Models

### Swallow Series

- **Maintainers**: Institute of Science Tokyo (formerly Tokyo Institute of Technology), AIST
- **Project**: https://swallow-llm.github.io/
- **License**: Llama 3.1 Community License (Meta)
- **Tested variant**: Llama-3.1-Swallow-8B-Instruct-v0.3
- **HuggingFace**: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3

### Llama 3.1

- **Maintainer**: Meta
- **License**: Llama 3.1 Community License
- **URL**: https://llama.meta.com/

## Llama License Obligations

When distributing model weights derived from Llama 3.1:

1. Include a copy of the Llama 3.1 license
2. Display "Built with Llama" in related materials
3. Do not use the model to improve non-Llama large language models
4. If monthly active users exceed 700M, request a separate license from Meta

Full license: https://llama.meta.com/llama3_1/license/

## Fine-tuned Adapters

Any LoRA adapters produced by `training/train_qlora.py` are derivative
works of the base model and inherit its license terms. The training code
itself is Apache 2.0.

## Adding a New Base

nllm is designed to be base-model-agnostic. To add support for a new base:

1. Place the GGUF file in `models/`
2. Create a Modelfile in `deploy/` or `models/`
3. Register the SHA-256 hash in `models/verify.sh`
4. Document provenance in this file
