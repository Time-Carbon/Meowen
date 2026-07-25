import os
from typing import Dict

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

### Global data
max_SFT_context = 1


### Dataset process


def make_CPT_conversation(dataset, tokenizer) -> Dict[str, str]:

    prompt = dataset["text"]

    global max_SFT_context
    if max_SFT_context < len(prompt):
        max_SFT_context = len(prompt)

    return {"text": prompt}


def make_SFT_conversation(dataset, tokenizer) -> Dict[str, str]:

    prompts = dataset["messages"]
    conversation = []

    for prompt in prompts:
        if prompt["role"] != "assistant":
            conversation.append({"role": prompt["role"], "content": prompt["content"]})
        else:
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"<think>\n{prompt["reasoning"]}\n</think>\n\n{prompt["content"]}",
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
        "--dataset_path", type=str, required=True, help="SFT 数据集路径"
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

    # 训练超参数
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA 秩 (rank)")
    parser.add_argument(
        "--lora_scale", type=float, default=1.0, help="等价于LoRA Alpha / LoRA Rank"
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=1.0, help="训练轮数 (epoch)"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05, help="模型热身步数占比"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率 (learning rate)")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="权重衰减正则化系数"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="AdamW优化器的Beta1参数"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="AdamW优化器的Beta2参数 (慎调)"
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_steps", type=int, default=400)
    parser.add_argument("--patience", type=int, default=0, help="早停 patience")
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
    parser.add_argument(
        "--cuda_um", action="store_true", help="是否使用CUDA统一内存进行训练"
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="验证集划分比例")

    return parser.parse_args()


### Main function
if __name__ == "__main__":
    args = get_args()

    import unsloth_model as um

    model, tokenizer = um.import_model(args.model_path, 0.95, False)

    if tokenizer.pad_token == "<|PAD_TOKEN|>" and args.continue_pretrain is True:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.lora is False:
        target_modules = [
            ### Attention
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            ### The last attention layer of Qwen3.5
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            ### MLP
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora = um.create_lora(model, args.lora_rank, args.lora_scale, target_modules)
    else:
        lora = model

    if args.continue_pretrain is False:
        dataset = um.load_data(
            path=args.dataset_path,
            tokenizer=tokenizer,
            load_from_cache=False,
            cache_dir=args.cache_dir,
            mapper_func=make_SFT_conversation,
        )
    else:
        dataset = um.load_data(
            path=args.dataset_path,
            tokenizer=tokenizer,
            load_from_cache=False,
            cache_dir=args.cache_dir,
            mapper_func=make_CPT_conversation,
        )

    dataset = dataset.train_test_split(test_size=args.test_size, shuffle=False)

    um.modelTrainer(
        cpt=args.continue_pretrain,
        resume=args.resume,
        threshold=args.threshold,
        patience=args.patience,
        lora=lora,
        tokenizer=tokenizer,
        dataset=dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_SFT_context=max_SFT_context,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=int(2 * args.per_device_eval_batch_size),
        max_grad_norm=args.max_grad_norm,
        output_dir=args.lora_cache_path,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_ratio,
    )
    um.save_lora(lora, tokenizer, args.lora_path)
