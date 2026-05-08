import os
from trl import GRPOTrainer, GRPOConfig


def GRPOtrain(
    lora,
    tokenizer,
    dataset,
    lr,
    steps,
    regularization,
    batch,
    reward_func,
    lora_path,
    max_RL_context,
):

    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

    generation_steps = 4 if batch < 4 else batch

    train_args = GRPOConfig(
        per_device_train_batch_size=2,
        num_generations=2,
        gradient_accumulation_steps=int(generation_steps / 2),
        steps_per_generation=int(generation_steps / 2),
        learning_rate=lr,
        lr_scheduler_type="cosine",
        max_steps=steps,
        warmup_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        logging_steps=1,
        max_completion_length=max_RL_context,
        temperature=1.0,
        top_p=0.95,
        loss_type="dr_grpo",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        output_dir=lora_path,
    )

    trainer = GRPOTrainer(
        args=train_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        model=lora,
        reward_processing_classes=tokenizer,
    )

    trainer.train()

    return lora
