"""LoRA SFT on filtered BOHDI traces."""

import argparse
import yaml

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_tokenizer = None


def format_example(messages, response):
    conversation = list(messages)
    conversation.append({"role": "assistant", "content": response})
    return _tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )


def add_text_column(batch):
    return {
        "text": [
            format_example(messages, response)
            for messages, response in zip(batch["messages"], batch["response"])
        ]
    }


def find_response_template(tokenizer):
    """Detect the assistant turn header by comparing templates with/without generation prompt.

    The difference between add_generation_prompt=True and False is exactly
    the assistant turn header (e.g. "<start_of_turn>model\\n" for Gemma,
    "<|start_header_id|>assistant<|end_header_id|>\\n\\n" for Llama 3).
    """
    dummy = [{"role": "user", "content": "hi"}]
    without_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
    with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)

    if with_gen.startswith(without_gen):
        template = with_gen[len(without_gen):]
        if template.strip():
            return template

    raise ValueError(
        f"Could not auto-detect response template. "
        f"Template suffix: {with_gen[-50:]!r}"
    )


def main():
    global _tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    _tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = DTYPE_MAP.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    print(f"Loading {model_cfg['name']} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], torch_dtype=dtype, device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )

    ds = load_dataset("json", data_files={"train": data_cfg["train_file"], "validation": data_cfg["val_file"]})
    ds = ds.map(add_text_column, batched=True, desc="Formatting chat transcripts")
    print(f"Train: {len(ds['train'])}  Val: {len(ds['validation'])}")

    # only compute loss on the assistant response, not on the prompt tokens
    response_template = find_response_template(_tokenizer)
    print(f"Response template for masking: {response_template!r}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=_tokenizer,
    )

    training_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"],
        bf16=train_cfg.get("bf16", True),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        gradient_checkpointing=True,
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=_tokenizer,
        data_collator=collator,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model("checkpoints/best")
    _tokenizer.save_pretrained("checkpoints/best")
    print("saved to checkpoints/best")


if __name__ == "__main__":
    main()
