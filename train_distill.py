"""Fine-tune a local model on the distilled dataset (LoRA SFT).

This is the training half of the distillation loop. It needs heavy
dependencies (torch, transformers, peft, datasets) that are intentionally
NOT in requirements.txt — run it on a GPU machine, Colab, or Kaggle, not on
the 2 CPU / 2 GB target box. The result is a LoRA adapter that can be merged
and converted to GGUF, then dropped into models/ and used as the local
chat model (engine.local), reducing dependence on the external provider.

Typical flow:
    python export_training_data.py
    python train_distill.py --dataset ./data/distill_dataset.jsonl
    # then merge + convert to GGUF (instructions printed at the end)
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on Napi's distilled dialogue dataset."
    )
    parser.add_argument("--dataset", default="./data/distill_dataset.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF base model to train (small models suit 2 CPU / 2 GB)")
    parser.add_argument("--output", default="./data/distill_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        import torch
    except ImportError as exc:
        print(f"Missing training dependency: {exc.name}")
        print("Install on a GPU machine or Colab/Kaggle:")
        print("  pip install torch transformers peft datasets")
        return 1

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING: CUDA not available. Training on CPU will be very slow; "
              "consider Colab/Kaggle or a GPU machine.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=args.max_length)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="epoch",
            fp16=use_cuda,
            report_to=[],
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nLoRA adapter saved to: {args.output}")
    print("To ship it as Napi's local model:")
    print("  1. Merge the adapter into the base model (peft merge_and_unload).")
    print("  2. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py <merged>")
    print("  3. Quantize:       llama.cpp/llama-quantize <model.gguf> <model-q4.gguf> Q4_K_M")
    print("  4. Move the .gguf into models/ and set engine.local in config.yaml:")
    print("       enabled: true")
    print("       model_path: ./models/<model-q4.gguf>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
