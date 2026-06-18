from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback


def modelTrainer(
    cpt,
    lora,
    tokenizer,
    dataset,
    lr,
    regularization,
    max_SFT_context,
    output_dir,
    resume,
    threshold,
    patience,
    eval_steps,
    **kwarg,
):

    train_args = SFTConfig(
        learning_rate=lr,
        warmup_steps=0.05,
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 0.1 * lr, "num_cycles": 1.5 if cpt else 0.5},
        max_length=max_SFT_context,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=int(2 * patience) if patience !=0 else None,
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

    if patience != 0:
        early_stop = EarlyStoppingCallback(
            early_stopping_patience=patience, early_stopping_threshold=threshold
        )
        trainer.add_callback(early_stop)

    trainer.train(resume_from_checkpoint=resume)

    return lora
