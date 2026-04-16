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


def format_example(batch):
    texts = []
    for msgs, resp in zip(batch["messages"], batch["response"]):
        msgs = list(msgs)
        msgs.append({"role": "assistant", "content": resp})
        texts.append(_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
    return texts


def find_response_template(tokenizer):
    """Figure out the token sequence that marks the start of an assistant turn.

    We render a tiny dummy conversation through the chat template and grab
    the text that appears right before the assistant's content.
    """
    dummy = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
    ]
    rendered = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)
    # find where the actual response content starts
    marker_end = rendered.rfind("Y")
    # walk backwards past whitespace to find the template boundary
    prefix = rendered[:marker_end]
    # grab the last line / tag before the response — that's our template
    for candidate_len in (30, 20, 10):
        candidate = prefix[-candidate_len:].lstrip()
        if candidate:
            return candidate
    # fallback: use a reasonably safe suffix
    return prefix[-15:]


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
        max_seq_length=train_cfg.get("max_seq_length", 4096),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=_tokenizer,
        data_collator=collator,
        formatting_func=format_example,
    )

    trainer.train()
    trainer.save_model("checkpoints/best")
    _tokenizer.save_pretrained("checkpoints/best")
    print("saved to checkpoints/best")


if __name__ == "__main__":
    main()
