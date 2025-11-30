import os
from dataclasses import dataclass, field
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class Args:
    model_id: str = field(default=os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    dataset_path: str = field(default="data/merged_dataset")
    output_dir: str = field(default="outputs/llama2-dapt")
    context_length: int = field(default=int(os.getenv("CTX_LEN", "1024")))
    lr: float = field(default=float(os.getenv("LR", "2e-4")))
    epochs: int = field(default=int(os.getenv("EPOCHS", "2")))
    train_batch_size: int = field(default=int(os.getenv("TRAIN_BS", "2")))
    eval_batch_size: int = field(default=int(os.getenv("EVAL_BS", "2")))
    gradient_accum: int = field(default=int(os.getenv("GRAD_ACCUM", "8")))
    warmup_ratio: float = field(default=float(os.getenv("WARMUP", "0.03")))
    save_steps: int = field(default=int(os.getenv("SAVE_STEPS", "500")))
    logging_steps: int = field(default=int(os.getenv("LOG_STEPS", "20")))
    eval_steps: int = field(default=int(os.getenv("EVAL_STEPS", "200")))
    lora_r: int = field(default=int(os.getenv("LORA_R", "16")))
    lora_alpha: int = field(default=int(os.getenv("LORA_ALPHA", "32")))
    lora_dropout: float = field(default=float(os.getenv("LORA_DROPOUT", "0.05")))
    load_in_4bit: bool = field(default=os.getenv("LOAD_IN_4BIT", "true").lower() == "true")
    load_in_8bit: bool = field(default=os.getenv("LOAD_IN_8BIT", "false").lower() == "true")

def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(examples["text"], truncation=True, max_length=max_len, padding=False)

def main():
    a = Args()
    if not torch.cuda.is_available():
        a.load_in_4bit = False
        a.load_in_8bit = False
    os.makedirs(a.output_dir, exist_ok=True)

    print("Loading dataset from:", a.dataset_path)
    dsd = load_from_disk(a.dataset_path)

    print("Loading tokenizer:", a.model_id)
    tokenizer = AutoTokenizer.from_pretrained(a.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing…")
    tokenized = dsd.map(
        lambda x: tokenize_function(x, tokenizer, a.context_length),
        batched=True,
        remove_columns=dsd["train"].column_names,
    )

    bnb_config = None
    if a.load_in_4bit or a.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=a.load_in_8bit,
            load_in_4bit=a.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    print("Loading base model:", a.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    if a.load_in_4bit or a.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.train_batch_size,
        per_device_eval_batch_size=a.eval_batch_size,
        gradient_accumulation_steps=a.gradient_accum,
        learning_rate=a.lr,
        warmup_ratio=a.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=a.logging_steps,
        evaluation_strategy="steps",
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=["none"],
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training…")
    trainer.train()
    print("Training complete.")

    print("Saving LoRA adapter to:", a.output_dir)
    trainer.save_model(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
