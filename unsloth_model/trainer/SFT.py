from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback


def SFTtrain(
    lora,
    tokenizer,
    dataset,
    lr,
    regularization,
    max_SFT_context,
    output_dir,
    resume,
    **kwarg,
):

    train_args = SFTConfig(
        warmup_ratio=0.05,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 2,
            "threshold": 0.01,
            "threshold_mode":"abs",
            "min_lr":1e-8
        },
        max_length=max_SFT_context,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,
        torch_empty_cache_steps=4,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
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

    trainer.train(resume_from_checkpoint=resume)

    return lora
