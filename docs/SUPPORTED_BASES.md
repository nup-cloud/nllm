# Supported Base Models

nllm is model-agnostic. It works with any GGUF-compatible large language model.
The runtime itself contains no model weights -- you bring your own base.

## Tested Bases

| Model | Parameters | Quantization | Notes |
|-------|-----------|-------------|-------|
| Swallow 8B Instruct v0.3 | 8B | Q4_K_M | Primary test target for Japanese IoT |
| Llama 3.1 8B Instruct | 8B | Q4_K_M | English-centric, works for EN commands |

## How to Add a New Base

1. **Place the GGUF file** in `models/` (or any local path).

2. **Create a Modelfile.** Copy `models/Modelfile.nllm` or `deploy/Modelfile` as a
   starting point. Adjust the `FROM` path and system prompt as needed.

   ```
   FROM ./models/your-model.gguf
   PARAMETER temperature 0.3
   SYSTEM """You are an IoT command generator..."""
   ```

3. **Register the SHA-256 hash** in `models/verify.sh` so integrity checks pass:

   ```bash
   shasum -a 256 models/your-model.gguf
   # Add the hash to the CHECKSUMS array in models/verify.sh
   ```

4. **Register with Ollama** (optional):

   ```bash
   ollama create your-model-name -f models/Modelfile.your-model
   ```

5. **Document provenance** in [MODEL_ORIGIN.md](../MODEL_ORIGIN.md) with license
   details for the upstream model.

## Notes

- `deploy/Modelfile` is an example configuration, not the only option.
  You can create multiple Modelfiles for different bases or use cases.
- Quantization level affects quality and speed. Q4_K_M is a good default
  for 8B-class models on consumer hardware.
- See [MODEL_ORIGIN.md](../MODEL_ORIGIN.md) for license obligations of
  tested base models.
