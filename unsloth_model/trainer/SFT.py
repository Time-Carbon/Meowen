from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported


def SFTtrain(
    lora, tokenizer, dataset, lr, regularization, batch, max_SFT_context, **kwarg
):

    train_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=int(batch / 2),
        warmup_ratio=0.05,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="cosine",
        max_length=max_SFT_context,
        max_grad_norm=1.0,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=int(batch / 2),
        eval_strategy="steps",
        eval_steps=4,
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

    trainer.train()

    return lora
