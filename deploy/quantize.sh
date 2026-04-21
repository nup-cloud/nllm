#!/usr/bin/env bash
# Quantize Swallow model to GGUF format for edge deployment
#
# Usage:
#   ./quantize.sh <input_model_dir> <output_dir> [quant_type]
#
# Examples:
#   ./quantize.sh ./merged_model ./output Q4_K_M
#   ./quantize.sh ./merged_model ./output Q2_K

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_model_dir> <output_dir> [quant_type]}"
OUTPUT_DIR="${2:?Usage: $0 <input_model_dir> <output_dir> [quant_type]}"
QUANT_TYPE="${3:-Q4_K_M}"

MODEL_NAME="nllm"

echo "=== nllm Model Quantization ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Type:   ${QUANT_TYPE}"
echo ""

# Check for llama.cpp
if ! command -v llama-quantize &>/dev/null; then
    echo "llama-quantize not found. Building llama.cpp..."
    if [[ ! -d "llama.cpp" ]]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    cd llama.cpp && make -j "$(nproc)" && cd ..
    QUANTIZE="./llama.cpp/llama-quantize"
else
    QUANTIZE="llama-quantize"
fi

mkdir -p "${OUTPUT_DIR}"

# Step 1: Convert to GGUF (if not already)
GGUF_F16="${OUTPUT_DIR}/${MODEL_NAME}-f16.gguf"
if [[ ! -f "${GGUF_F16}" ]]; then
    echo "[1/2] Converting to GGUF FP16..."
    python3 -c "
from llama_cpp import Llama
print('Using llama-cpp-python for conversion reference')
" 2>/dev/null || true

    # Use llama.cpp converter
    if [[ -f "llama.cpp/convert_hf_to_gguf.py" ]]; then
        python3 llama.cpp/convert_hf_to_gguf.py "${INPUT_DIR}" --outfile "${GGUF_F16}"
    else
        echo "ERROR: llama.cpp/convert_hf_to_gguf.py not found"
        echo "Please convert to GGUF manually first."
        exit 1
    fi
else
    echo "[1/2] GGUF FP16 already exists, skipping conversion."
fi

# Step 2: Quantize
GGUF_QUANT="${OUTPUT_DIR}/${MODEL_NAME}-${QUANT_TYPE,,}.gguf"
echo "[2/2] Quantizing to ${QUANT_TYPE}..."
${QUANTIZE} "${GGUF_F16}" "${GGUF_QUANT}" "${QUANT_TYPE}"

echo ""
echo "=== Done ==="
echo "Output: ${GGUF_QUANT}"
echo "Size:   $(du -h "${GGUF_QUANT}" | cut -f1)"
echo ""
echo "To use with Ollama:"
echo "  cp ${GGUF_QUANT} models/"
echo "  ollama create ${MODEL_NAME} -f models/Modelfile.swallow"
echo "  ollama run ${MODEL_NAME}"
