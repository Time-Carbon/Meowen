from unsloth import FastLanguageModel


def import_model(
    model_path: str,
    mem_usage: float,
    load_to_vllm: bool = False,
    max_RL_context: int = 4096,
):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_RL_context,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=True,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=mem_usage,
        fast_inference=load_to_vllm,
    )

    return model, tokenizer
