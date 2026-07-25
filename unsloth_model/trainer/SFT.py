from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
from typing import Any
from datasets import DatasetDict


def modelTrainer(
    cpt: bool,
    lora: bool,
    tokenizer: Any,
    dataset: DatasetDict | Any,
    lr: float,
    max_SFT_context: int,
    output_dir: str,
    resume: bool,
    threshold: float,
    patience: int,
    eval_steps: int,
    **kwarg,
):

    train_args = SFTConfig(
        learning_rate=lr,
        bf16=is_bfloat16_supported(),
        bf16_full_eval=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        fp16_full_eval=not is_bfloat16_supported(),
        logging_steps=1,
        torch_empty_cache_steps=8,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.01},
        max_length=max_SFT_context,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=int(2 * patience) if patience != 0 else None,
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
