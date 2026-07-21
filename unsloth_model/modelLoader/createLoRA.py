from unsloth import FastLanguageModel
from typing import Any, List


def create_lora(model: Any, rank: int, scale: float, target_modules: List[str]):

    lora = FastLanguageModel.get_peft_model(
        model=model,
        r=rank,
        lora_alpha=int(scale * rank),
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        target_modules=target_modules,
    )

    return lora
