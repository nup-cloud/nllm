"""Evaluation script for fine-tuned IoT LLM.

Measures command generation accuracy against test data.

Usage:
    python eval.py \
        --model output/lora_adapter \
        --test-data data/iot_commands_ja.jsonl \
        --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_command_name(output: str) -> str:
    """Extract the main command name from structured output."""
    match = re.match(r"(\w+)\(", output.strip())
    return match.group(1) if match else output.strip().split("(")[0]


def evaluate_exact_match(predictions: list[str], references: list[str]) -> float:
    """Calculate exact match accuracy."""
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return correct / len(references) if references else 0.0


def evaluate_command_match(predictions: list[str], references: list[str]) -> float:
    """Calculate command-name-level accuracy (ignoring parameters)."""
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_cmd = extract_command_name(pred)
        ref_cmd = extract_command_name(ref)
        if pred_cmd == ref_cmd:
            correct += 1
    return correct / len(references) if references else 0.0


def load_test_data(filepath: str, num_samples: int | None = None) -> list[dict]:
    """Load test samples from JSONL."""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if num_samples:
        samples = samples[:num_samples]
    return samples


def load_predictions(filepath: str) -> list[dict]:
    """Load predictions from a JSONL file.

    Each line should have at minimum: {"instruction": "...", "output": "...", "prediction": "..."}
    """
    predictions = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def print_results(
    test_data: list[dict],
    predictions: list[str],
    references: list[str],
) -> None:
    """Print evaluation metrics and sample predictions."""
    exact = evaluate_exact_match(predictions, references)
    cmd = evaluate_command_match(predictions, references)

    print("\n=== Evaluation Results ===")
    print(f"Samples:           {len(references)}")
    print(f"Exact Match:       {exact:.1%}")
    print(f"Command Match:     {cmd:.1%}")
    print()

    # Show some examples
    print("=== Sample Predictions ===")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}] Input:     {test_data[i]['instruction']}")
        print(f"    Expected:  {references[i]}")
        print(f"    Predicted: {predictions[i]}")
        print(f"    Match:     {'OK' if predictions[i] == references[i] else 'MISMATCH'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IoT LLM")
    parser.add_argument("--model", default=None, help="Path to fine-tuned model")
    parser.add_argument("--test-data", required=True, help="Path to test JSONL file")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--base-model", default=None, help="Base model for PEFT adapter loading")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: skip model loading, read predictions from --predictions file",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions JSONL file (required with --offline). "
        "Each line: {\"instruction\": ..., \"output\": ..., \"prediction\": ...}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_data = load_test_data(args.test_data, args.num_samples)
    logger.info("Loaded %d test samples", len(test_data))

    # --- Offline mode: read predictions from file ---
    if args.offline:
        if not args.predictions:
            parser.error("--predictions is required when using --offline")
        logger.info("Offline mode: reading predictions from %s", args.predictions)
        pred_data = load_predictions(args.predictions)
        logger.info("Loaded %d predictions", len(pred_data))

        # Build lookup from instruction -> prediction
        pred_lookup = {p["instruction"]: p["prediction"] for p in pred_data}

        predictions = []
        references = []
        skipped = 0
        for sample in test_data:
            instruction = sample["instruction"]
            if instruction in pred_lookup:
                predictions.append(pred_lookup[instruction].strip())
                references.append(sample["output"])
            else:
                skipped += 1

        if skipped:
            logger.warning(
                "Skipped %d test samples with no matching prediction", skipped
            )

        print_results(test_data, predictions, references)
        return

    # --- Online mode: load model and generate predictions ---
    if not args.model:
        parser.error("--model is required when not using --offline")

    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        return

    logger.info("Loading model from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    predictions = []
    references = []

    for i, sample in enumerate(test_data):
        instruction = sample["instruction"]
        expected = sample["output"]

        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "あなたは国産IoT特化AIアシスタントです。"
            "ユーザーの日本語指示をIoTデバイス制御コマンドに変換してください。"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
            )
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predictions.append(generated.strip())
        references.append(expected)

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d / %d", i + 1, len(test_data))

    print_results(test_data, predictions, references)


if __name__ == "__main__":
    main()
