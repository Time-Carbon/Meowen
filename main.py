### Unsloth related
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch

### Dataset related
from datasets import load_dataset
import re

### Train related
from trl import SFTTrainer
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward, think_format_reward
from transformers import TrainingArguments

### Global data
max_context: int = 16384
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
def think_reward(completions):

    format_rewards = think_format_reward(completions)
    length_rewards = []

    for completion in completions:
        content = completion[0]["content"]
        think_content = re.match("^<think>(.*?)</think>", content, re.DOTALL)
        response_content = content[len(think_content) :]

        if len(response_content) == 0 or len(think_content) == 0:
            length_rate = 0.1
        else:
            length_rate = len(think_content) / len(response_content)

        length_reward = 0
        if length_rate > 0 and length_rate <= 2:
            ### y = -(1/4)(x^2) + x
            length_reward = -0.25 * (length_rate**2) + length_rate
        length_rewards.append(length_reward)

    rewards = [0.5 * (x + y) for x, y in zip(format_rewards, length_reward)]

    return rewards


def keyword_reward(completions):

    rewards = []

    return rewards


def reward_func(completions, answer, **kwargs):

    rewards = []

    accuracy_rewards = accuracy_reward(completions, answer)
    think_rewards = think_reward(completions)

    rewards = [
        0.5 * (accuracy + think)
        for accuracy, think in zip(accuracy_rewards, think_rewards)
    ]

    return rewards


### Main process
def import_model():

    qwen_template = "\
        \{\%\ for message in messages \%\}\
            \{\%\ if message['role'] == 'system' \%\}\
                \{\{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' \}\}\
            \{\%\ elif message['role'] == 'user' \%\}\
                \{\{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' \}\}\
            \{\%\ elif message['role'] == 'assistant' \%\}\
                \{\{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' \}\}\
            \{\%\ endif \%\}\
        \{\%\ endfor \%\}\
        \{\%\ if add_generation_prompt \%\}\
            \{\{ '<|im_start|>assistant\n' \}\}\
        \{\%\ endif \%\}\
    "

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

    tokenizer.chat_template = qwen_template.strip()

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


def make_RL_conversation(dataset, tokenizer):

    system_prompt = r"你的任务是解决`user`提出的问题，要求讲解答题思路，并将最终答案输出至$\box{}$中"

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dataset["question"] + "\n" + dataset["options"]},
    ]

    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    return {"prompt": prompt, "answer": dataset["answer"]}


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


def GRPOtrain(lora, tokenizer, dataset, lr, steps, regularization):

    train_args = GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        max_steps=steps,
        warmup_steps=0.1 * steps,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        logging_steps=0.1 * steps,
        max_completion_length=max_context,
        temperature=1.0,
        top_p=0.95,
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


def save_lora(lora, tokenizer):

    lora.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


### Main function
if __name__ == "__main__":

    model, tokenizer = import_model()

    lora = create_lora(model, 8)

    rl_dataset, sft_dataset, keyword = load_data()

    rl_dataset = rl_dataset.map(
        function=make_RL_conversation, fn_kwargs={"tokenizer": tokenizer}
    )

    GRPOtrain(
        lora=lora,
        tokenizer=tokenizer,
        dataset=rl_dataset,
        steps=300,
        regularization=0.01,
        lr=5e-6,
    )
