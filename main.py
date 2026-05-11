import unsloth_model as um
import os

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"


### Global data
max_SFT_context = 1
model_path = "./qwen_model/qwen3.5/0.8B_base_8bit/"
lora_path = "./lora/"
dataset_path = "./dataset/"
sft_dataset_path = dataset_path + "raw_text/"
cache_dir = dataset_path + "cache/"


### Dataset process


def make_SFT_conversation(dataset, tokenizer):

    prompt = dataset["text"]

    global max_SFT_context
    if max_SFT_context < len(prompt):
        max_SFT_context = len(prompt)

    return {"text": prompt}


### Main function
if __name__ == "__main__":

    model, tokenizer = um.import_model(model_path, 0.95, False)

    lora = um.create_lora(model, 32)

    sft_dataset = um.load_data(
        path=sft_dataset_path,
        tokenizer=tokenizer,
        load_from_cache=True,
        cache_dir=cache_dir,
        mapper_func=make_SFT_conversation,
    )

    sft_dataset = sft_dataset.train_test_split(test_size=0.1,shuffle=True)

    um.SFTtrain(
        lora=lora,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        lr=1e-5,
        regularization=1e-5,
        max_SFT_context=max_SFT_context,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_grad_norm=8,
    )
    um.save_lora(lora, tokenizer, lora_path)
