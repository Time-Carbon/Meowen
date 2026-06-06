import os

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

### Global data
max_SFT_context = 1


### Dataset process


def make_CPT_conversation(dataset, tokenizer):

    prompt = dataset["text"]

    global max_SFT_context
    if max_SFT_context < len(prompt):
        max_SFT_context = len(prompt)

    return {"text": prompt}


def make_SFT_conversation(dataset, tokenizer):

    prompts = dataset["messages"]
    conversation = []

    for prompt in prompts:
        if prompt["role"] != "assistant":
            conversation.append(
                {
                    "role": prompt["role"],
                    "content": prompt["content"]
                }
            )
        else:
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"<think>\n{prompt["reasoning"]}\n</think>\n\n{prompt["content"]}"
                }
            )

    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )

    global max_SFT_context
    if max_SFT_context < len(text):
        max_SFT_context = len(text)

    return {"text": text}


### Get args
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="SFT LoRA training script")

    # 路径参数
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument(
        "--sft_dataset_path", type=str, required=True, help="SFT 数据集路径"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="数据集缓存目录"
    )
    parser.add_argument(
        "--lora_cache_path",
        type=str,
        default="./lora_cache",
        help="训练中间 checkpoint 输出目录",
    )
    parser.add_argument(
        "--lora_path", type=str, default="./lora_model", help="最终保存 LoRA 权重的路径"
    )

    # 训练超参数（题目要求重点提取的 lora_rank, epoch, learn_rate）
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA 秩 (rank)")
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="训练轮数 (epoch)"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率 (learning rate)")

    # 其他常用训练参数
    parser.add_argument(
        "--regularization", type=float, default=1e-4, help="权重衰减正则化系数"
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--patience", type=int, default=3, help="早停 patience")
    parser.add_argument("--threshold", type=float, default=1e-3, help="早停阈值")
    parser.add_argument(
        "--resume", action="store_true", help="是否从 checkpoint 恢复训练"
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="加载的模型是否是已训练好的 LoRA 模型",
    )
    parser.add_argument(
        "--continue_pretrain", action="store_true", help="是否进行CPT (继续预训练) "
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="验证集划分比例")

    return parser.parse_args()


### Main function
if __name__ == "__main__":
    args = get_args()

    import unsloth_model as um

    model, tokenizer = um.import_model(args.model_path, 0.95, False)

    if tokenizer.pad_token == "<|PAD_TOKEN|>":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.lora == False:
        lora = um.create_lora(model, args.lora_rank)
    else:
        lora = model

    if args.continue_pretrain == False:
        sft_dataset = um.load_data(
            path=args.sft_dataset_path,
            tokenizer=tokenizer,
            load_from_cache=False,
            cache_dir=args.cache_dir,
            mapper_func=make_SFT_conversation,
        )
    else:
        sft_dataset = um.load_data(
            path=args.sft_dataset_path,
            tokenizer=tokenizer,
            load_from_cache=False,
            cache_dir=args.cache_dir,
            mapper_func=make_CPT_conversation,
        )

    sft_dataset = sft_dataset.train_test_split(test_size=args.test_size, shuffle=False)

    um.SFTtrain(
        resume=args.resume,
        threshold=args.threshold,
        patience=args.patience,
        lora=lora,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        lr=args.lr,
        regularization=args.regularization,
        max_SFT_context=max_SFT_context,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.lora_cache_path,
    )
    um.save_lora(lora, tokenizer, args.lora_path)
