### Unsloth related
import unsloth
import torch

### Dataset related
import re

### Train related
from trl.rewards import accuracy_reward, think_format_reward

### Arg related
import argparse

import unsloth_model as um

### Global data
max_RL_context = 4096
max_SFT_context = 1
model_path = "./qwen_model/qwen3/1.7B_Base_8bit/"
lora_path = "./lora/"
dataset_path = "./dataset/"
sft_dataset_path = dataset_path + "sft/"
rl_dataset_path = dataset_path + "rl/"
keyword_dataset_path = dataset_path + "keyword/"


### Dataset process
def make_RL_conversation(dataset, tokenizer):

    system_prompt = r"\
        你的任务是解决`user`提出的问题，先在<think></think>中思考，然后用猫娘的语气回答。\
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
        prompt, tokenize=False, add_generation_prompt=False
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
        think_content = re.match("^<think>(.*?)</think>", content, re.DOTALL)
        response_content = ""

        ### Must have thinking
        if think_content == None:
            think_content = ""
        else:
            think_content = think_content.group()
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


### Main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select train mode")
    parser.add_argument("-m", "--mode", type=str, help="Supported mode: sft, rl")
    args = parser.parse_args()

    if args.mode == "sft":

        model, tokenizer = um.import_model(model_path, 0.95, False)

        lora = um.create_lora(model, 16)

        sft_dataset = um.load_data(
            sft_dataset_path, tokenizer, True, sft_dataset_path, make_SFT_conversation
        )

        um.SFTtrain(
            lora=lora,
            tokenizer=tokenizer,
            dataset=sft_dataset,
            steps=100,
            lr=5e-4,
            regularization=1e-2,
            batch=8,
            max_SFT_context=max_SFT_context,
        )
        um.save_lora(lora, tokenizer, lora_path)

    elif args.mode == "rl":

        model, tokenizer = um.import_model(model_path, 0.5, True)

        lora = um.create_lora(model, 8)
        rl_dataset = um.load_data(
            rl_dataset_path, tokenizer, True, rl_dataset_path, make_RL_conversation
        )

        um.GRPOtrain(
            lora=lora,
            tokenizer=tokenizer,
            dataset=rl_dataset,
            steps=300,
            regularization=0.01,
            lr=1e-4,
            batch=4,
        )
        um.save_lora(lora, tokenizer, lora_path)
        torch.distributed.destroy_process_group()
