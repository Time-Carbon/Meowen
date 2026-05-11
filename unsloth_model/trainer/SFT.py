from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback


def SFTtrain(
    lora, tokenizer, dataset, lr, regularization, max_SFT_context, output_dir, **kwarg
):

    train_args = SFTConfig(
        warmup_ratio=0.05,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="cosine",
        max_length=max_SFT_context,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,
        torch_empty_cache_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        output_dir=output_dir,
        greater_is_better=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        **kwarg,
    )

    trainer = SFTTrainer(
        model=lora,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=max_SFT_context,
        packing=False,
        args=train_args,
    )

    early_stop = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.01
    )

    trainer.add_callback(early_stop)

    trainer.train()

    return lora
