from unsloth import FastLanguageModel


def create_lora(model, rank):

    lora = FastLanguageModel.get_peft_model(
        model=model,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=[
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return lora
