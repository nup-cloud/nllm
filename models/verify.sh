#!/usr/bin/env bash
# SHA-256 integrity verification for model files
# Usage:
#   ./verify.sh check     — verify downloaded model integrity
#   ./verify.sh download  — download Swallow GGUF from HuggingFace

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}"

# Known checksums (update after each model release)
declare -A CHECKSUMS=(
    ["swallow-8b-instruct-q4_k_m.gguf"]="PLACEHOLDER_SHA256_UPDATE_AFTER_DOWNLOAD"
)

HF_REPO="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3-GGUF"

cmd_check() {
    echo "=== nllm Model Integrity Check ==="
    local all_ok=true

    for file in "${!CHECKSUMS[@]}"; do
        local filepath="${MODEL_DIR}/${file}"
        if [[ ! -f "$filepath" ]]; then
            echo "  MISSING: ${file}"
            all_ok=false
            continue
        fi

        local expected="${CHECKSUMS[$file]}"
        local actual
        actual=$(shasum -a 256 "$filepath" | awk '{print $1}')

        if [[ "$expected" == "PLACEHOLDER_SHA256_UPDATE_AFTER_DOWNLOAD" ]]; then
            echo "  SKIP: ${file} (checksum not yet registered)"
            echo "  Actual SHA-256: ${actual}"
            echo "  Update CHECKSUMS in this script with the above value."
        elif [[ "$actual" == "$expected" ]]; then
            echo "  OK: ${file}"
        else
            echo "  FAIL: ${file}"
            echo "    Expected: ${expected}"
            echo "    Actual:   ${actual}"
            all_ok=false
        fi
    done

    if $all_ok; then
        echo "All checks passed."
    else
        echo "Some checks failed. Re-download models if needed."
        exit 1
    fi
}

cmd_download() {
    echo "=== Downloading Swallow 8B GGUF ==="

    if ! command -v huggingface-cli &>/dev/null; then
        echo "Installing huggingface_hub..."
        pip install -q huggingface_hub
    fi

    echo "Downloading from ${HF_REPO}..."
    huggingface-cli download "${HF_REPO}" \
        --local-dir "${MODEL_DIR}" \
        --include "*.gguf"

    echo "Download complete. Run './verify.sh check' to verify integrity."
}

case "${1:-check}" in
    check)    cmd_check ;;
    download) cmd_download ;;
    *)
        echo "Usage: $0 {check|download}"
        exit 1
        ;;
esac
