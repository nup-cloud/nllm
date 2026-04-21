"""QLoRA fine-tuning script for Swallow IoT adaptation.

Designed to run on Google Colab free tier (T4 GPU).

Usage:
    python train_qlora.py \
        --base-model tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3 \
        --data-dir data/ \
        --output-dir output/ \
        --epochs 3

Requirements:
    pip install torch transformers peft datasets bitsandbytes accelerate trl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_training_data(data_dir: str) -> list[dict]:
    """Load all JSONL files from data directory."""
    data_path = Path(data_dir)
    all_samples = []

    for jsonl_file in sorted(data_path.glob("*.jsonl")):
        logger.info("Loading %s", jsonl_file.name)
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    all_samples.append(sample)

    logger.info("Total training samples: %d", len(all_samples))
    return all_samples


def format_prompt(sample: dict) -> str:
    """Format a training sample into Llama 3.1 chat format."""
    system = (
        "あなたは国産IoT特化AIアシスタントです。"
        "ユーザーの日本語指示をIoTデバイス制御コマンドに変換してください。"
    )
    instruction = sample["instruction"]
    output = sample["output"]

    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for IoT LLM")
    parser.add_argument("--base-model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--data-dir", default="data/", help="Directory containing JSONL training data")
    parser.add_argument("--output-dir", default="output/", help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Load data ---
    raw_data = load_training_data(args.data_dir)
    if not raw_data:
        logger.error("No training data found in %s", args.data_dir)
        return

    # --- Import ML libraries ---
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        logger.error(
            "Missing dependency: %s\n"
            "Install with: pip install torch transformers peft datasets bitsandbytes accelerate trl",
            e,
        )
        return

    # --- Quantization config (4-bit for T4 GPU) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Load base model ---
    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # --- Prepare dataset ---
    formatted = [{"text": format_prompt(s)} for s in raw_data]
    dataset = Dataset.from_list(formatted)
    logger.info("Dataset size: %d samples", len(dataset))

    # --- Training ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    training_args.max_seq_length = args.max_seq_len
    training_args.dataset_text_field = "text"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    logger.info("Starting QLoRA training...")
    trainer.train()

    # --- Save ---
    logger.info("Saving fine-tuned model to %s", output_dir)
    trainer.save_model(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    logger.info("Training complete! Merge adapter with base model for deployment.")
    logger.info("Next step: python -c \"from peft import AutoPeftModelForCausalLM; ...\"")


if __name__ == "__main__":
    main()
