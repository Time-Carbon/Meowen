### Unsloth related
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
import os

# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

### Dataset related
from datasets import load_dataset
import re

### Train related
from trl import SFTTrainer, SFTConfig
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward, think_format_reward

### Global data
max_RL_context = 8192
max_SFT_context = 1
model_path = "./model/qwen3.5/2B_base_8bit/"
lora_path = "./lora/"
dataset_path = "./dataset/"
sft_dataset_path = dataset_path + "sft/"
rl_dataset_path = dataset_path + "rl/"
keyword_dataset_path = dataset_path + "keyword/"


### Dataset process
def keyword_loader(dataset_path):

    kaomoji = load_dataset(
        path="kareudon/kaomoji-tagged",
        cache_dir=dataset_path,
        split="train",
    )
    emotion = load_dataset(path=dataset_path + "emotion/", split="train")

    keyword = {"kaomoji": kaomoji.to_dict, "emotion_word": emotion.to_dict}

    return keyword


def make_RL_conversation(dataset, tokenizer):

    system_prompt = r"\
        你的任务是解决`user`提出的问题，先在<think></think>中思考，后回答。\
        要求讲解答题思路，并将最终答案输出至$\box{}$中。\
        "

    prompt = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": dataset["question"] + "\n" + dataset["options"]},
    ]

    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    return {"prompt": prompt, "answer": dataset["answer"]}


def make_SFT_conversation(dataset, tokenizer):

    system_prompt = r"\
        你的任务是解决`user`提出的问题，先在<think></think>中思考，后回答。\
        "

    prompt = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": dataset["user"]},
        {
            "role": "assistant",
            "content": dataset["assistant"],
        },
    ]

    prompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    global max_SFT_context
    if max_SFT_context < len(prompt):
        max_SFT_context = len(prompt)

    return {"text": prompt}


### Reward functions
def think_reward(completions):

    format_rewards = think_format_reward(completions)
    length_rewards = []

    for completion in completions:
        content = completion[0]["content"]
        think_content = re.match("^<think>(.*?)</think>", content, re.DOTALL).group()
        response_content = ""

        ### Must have thinking
        if think_content == None:
            think_content = ""
        else:
            response_content = content[len(think_content) :]

        ### Must both have thinking and responding
        if len(response_content) == 0 or len(think_content) == 0:
            length_rate = 0.1
        else:
            length_rate = len(think_content) / len(response_content)

        length_reward = 0
        if length_rate > 0 and length_rate <= 2:
            ### y = -(1/4)(x^2) + x
            length_reward = -0.25 * (length_rate**2) + length_rate

        length_rewards.append(length_reward)

    rewards = [0.5 * (x + y) for x, y in zip(format_rewards, length_rewards)]

    return rewards


def language_reward(completions):

    rewards = []

    for completion in completions:
        total_len = len(completion[0]["content"])
        target_language_lenght = 0

        for char in completion[0]["content"]:
            if (
                char >= "\u4e00" and char <= "\u9fff"
            ):  ### compare char with the Unicode range of Chinese
                target_language_lenght += 1

        if total_len != 0:
            lenght_rate = target_language_lenght / total_len
            reward = -4 * (lenght_rate**2) + 4 * lenght_rate
        else:
            reward = 0.1

        rewards.append(reward)

    return rewards


def repetition_penalty(completions):

    penalty = []

    return penalty


def keyword_reward(completions):

    rewards = []

    return rewards


def reward_func(completions, answer, **kwargs):

    output = [[{"content": "<think>\n" + completion}] for completion in completions]

    accuracy_rewards = accuracy_reward(completions=output, solution=answer)
    think_rewards = think_reward(output)
    language_rewards = language_reward(output)

    rewards = [
        0.45 * (accuracy + think) + 0.1 * language
        for accuracy, think, language in zip(
            accuracy_rewards, think_rewards, language_rewards
        )
    ]

    return rewards


### Main process
def import_model(model_path, mem_usage, load_to_vllm):

    qwen_template = """
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' -}}
    {%- elif message['role'] == 'user' -%}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n<think>\n' -}}
{%- endif -%}
"""

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
        ### Train language model
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    return lora


def load_data(tokenizer, load_from_cache=False):

    rl = []
    sft = []
    keyword = []

    ### Load raw Dataset
    rl = load_dataset(
        path="ecnu-icalk/cmm-math", cache_dir=rl_dataset_path, split="train"
    )
    sft = load_dataset(path=sft_dataset_path, split="train")
    # keyword = keyword_loader(keyword_dataset_path)

    ### Remap RL dataset
    rl_dataset_col_name = rl.column_names
    rl = rl.map(
        make_RL_conversation,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=load_from_cache,
        remove_columns=rl_dataset_col_name,
    )

    ### Remap SFT dataset
    sft_dataset_col_name = sft.column_names
    sft = sft.map(
        make_SFT_conversation,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=load_from_cache,
        remove_columns=sft_dataset_col_name,
    )

    return rl, sft, keyword


def SFTtrain(lora, tokenizer, dataset, steps, lr, regularization, batch):

    train_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=int(batch / 2),
        warmup_steps=1,
        max_steps=steps,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=regularization,
        lr_scheduler_type="cosine",
        max_length=max_SFT_context,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=lora,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_SFT_context,
        packing=False,
        args=train_args,
    )

    trainer.train()

    return lora


def GRPOtrain(lora, tokenizer, dataset, lr, steps, regularization, batch):

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


def save_lora(lora, tokenizer):

    lora.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)


### Main function
if __name__ == "__main__":

    model, tokenizer = import_model(model_path, 0.95, False)

    lora = create_lora(model, 8)

    rl_dataset, sft_dataset, keyword = load_data(tokenizer, load_from_cache=True)

    SFTtrain(
        lora=lora,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        steps=60,
        lr=1e-4,
        regularization=1e-2,
        batch=8,
    )

    GRPOtrain(
        lora=lora,
        tokenizer=tokenizer,
        dataset=rl_dataset,
        steps=300,
        regularization=0.01,
        lr=1e-4,
        batch=4,
    )

    save_lora(lora, tokenizer)
    torch.distributed.destroy_process_group()
