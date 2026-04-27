### Unsloth related
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch

### Dataset related
from datasets import load_dataset

### Train related
from trl import SFTTrainer
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments

### Global data
max_context = 32768 / 2
model_path = "./model/2B_base_8bit/"
lora_path = "./lora/"
dataset_path = "./dataset/"


### Dataset process
def dataset_loader(dataset_path):

    dataset = load_dataset(
        path="ecnu-icalk/cmm-math", cache_dir=dataset_path + "dataset/", split="train"
    )

    return dataset


def sftdata_loader(dataset_path):

    sft = load_dataset(path=dataset_path + "sft/", split="train")

    return sft


def keyword_loader(dataset_path):

    kaomoji = load_dataset(
        path="kareudon/kaomoji-tagged",
        cache_dir=dataset_path + "keyword/",
        split="train",
    )
    emotion = load_dataset(path=dataset_path + "keyword/emotion/", split="train")

    keyword = {"kaomoji": kaomoji.to_dict, "emotion_word": emotion.to_dict}

    return keyword


### Reward functions
def reward_func():

    reward = 0

    return reward


### Main process
def import_model():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_context,
        dtype=None,
        load_in_8bit=True,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=0.8,
        fast_inference=False,
    )

    return model, tokenizer


def create_lora(model, rank):

    lora = FastLanguageModel.get_peft_model(
        model=model,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        ### Trained layers
        finetune_mlp_modules=True,
        finetune_attention_modules=True,
        finetune_language_layers=True,
        finetune_vision_layers=False,
    )

    return lora


def load_data():

    rl = dataset_loader(dataset_path)
    sft = sftdata_loader(dataset_path)
    keyword = keyword_loader(dataset_path)

    return rl, sft, keyword


def SFTtrain(lora, tokenizer, dataset, steps, lr, regularization):

    train_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=0.1 * steps,
        max_steps=steps,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=0.1 * steps,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="linear",
    )

    trainer = SFTTrainer(
        model=lora,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_context,
        packing=False,
        args=train_args,
    )

    trainer.train()

    return lora


def GRPOtrain(lora, dataset, reward_func, lr, steps, regularization):

    train_args = GRPOConfig(
        per_device_train_batch_size=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        max_steps=steps,
        warmup_steps=0.1 * steps,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
    )

    trainer = GRPOTrainer(
        args=train_args, train_dataset=dataset, reward_funcs=reward_func, model=lora
    )

    trainer.train()

    return lora


def save_lora(lora, tokenizer):

    lora.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


### Main function
if __name__ == "__main__":

    model, tokenizer = import_model()

    lora = create_lora(model, 8)

    rl_dataset, sft_dataset, keyword = load_data()
