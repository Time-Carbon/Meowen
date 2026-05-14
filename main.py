import unsloth_model as um
import os

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"


### Global data
max_SFT_context = 1
model_path = "./model/"
lora_path = "./lora/"
sft_dataset_path = "./dataset/"
cache_dir = "./cache/"
lora_cache_path = "./lora_cache/"


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

    lora = um.create_lora(model, 64)

    sft_dataset = um.load_data(
        path=sft_dataset_path,
        tokenizer=tokenizer,
        load_from_cache=True,
        cache_dir=cache_dir,
        mapper_func=make_SFT_conversation,
    )

    sft_dataset = sft_dataset.train_test_split(test_size=0.1, shuffle=False)

    um.SFTtrain(
        resume=False,
        lora=lora,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        lr=1e-4,
        regularization=1e-6,
        max_SFT_context=max_SFT_context,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        max_grad_norm=8,
        output_dir=lora_cache_path,
    )
    um.save_lora(lora, tokenizer, lora_path)
