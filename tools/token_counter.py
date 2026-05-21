#!/usr/bin/env python3
"""
统计 Parquet 数据集中所有文本分词后每个 token 的出现频次。
使用多进程加速分词，结果输出为 JSON 列表。
"""

import argparse
import json
import os
from collections import Counter
from glob import glob
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any

import pandas as pd
from transformers import AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="统计 Parquet 文本数据分词后的 token 频率。"
    )
    parser.add_argument(
        "--parquet_dir",
        required=True,
        help="包含 Parquet 文件的文件夹路径。",
    )
    parser.add_argument(
        "--tokenizer_dir",
        required=True,
        help="预训练 tokenizer 所在的文件夹路径。",
    )
    parser.add_argument(
        "--output",
        default="token_counts.json",
        help="输出 JSON 文件路径，默认为 token_counts.json。",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="Parquet 文件中文本列的名称，默认为 'text'。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行工作进程数，默认为 CPU 核心数。",
    )
    return parser.parse_args()


def load_parquet_file_list(parquet_dir: str) -> List[str]:
    """
    加载指定文件夹下所有 Parquet 文件的列表。

    参数:
        parquet_dir: Parquet 文件所在文件夹。

    返回:
        所有文件的列表。
    """
    pattern = os.path.join(parquet_dir, "*.parquet")
    file_paths = sorted(glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"在 {parquet_dir} 中未找到任何 .parquet 文件。")
    return file_paths


def load_parquet_text(file_paths: List[str], text_column: str = "text") -> List[str]:
    """
    加载指定文件夹下所有 Parquet 文件的文本列。

    参数:
        file_paths: Parquet 文件所在文件夹。
        text_column: 文本列的名称。

    返回:
        所有文本行的列表。
    """
    all_texts = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path, columns=[text_column])
        # 确保列存在
        if text_column not in df.columns:
            raise ValueError(f"文件 {file_path} 中未找到列 '{text_column}'。")
        # 过滤空值，转为字符串
        texts = df[text_column].dropna().astype(str).tolist()
        all_texts.extend(texts)

    if not all_texts:
        raise ValueError("所有 Parquet 文件中均无有效文本。")

    return all_texts


def load_tokenizer(tokenizer_dir: str) -> AutoTokenizer:
    """
    加载预训练分词器。

    参数:
        tokenizer_dir: 分词器文件夹路径。

    返回:
        AutoTokenizer 实例。
    """
    # 指定 use_fast=True 可启用快速分词器（Rust 实现），性能更好
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    return tokenizer


def tokenize_chunk(args: tuple) -> Counter:
    """
    对一批文本进行分词并统计 token 频次（供子进程调用）。

    参数:
        args: 一个元组 (chunk_texts, tokenizer_dir)。

    返回:
        本批次的 token 计数器。
    """
    chunk_file_list, tokenizer_dir, text_column = args
    tokenizer = load_tokenizer(tokenizer_dir)
    local_counter = Counter()

    texts = load_parquet_text(chunk_file_list, text_column)

    for text in texts:
        # 仅获取 token 字符串，不映射为 ID
        tokens = tokenizer.tokenize(text)
        local_counter.update(tokens)

    return local_counter


def merge_counters(counters: List[Counter]) -> Counter:
    """
    合并多个计数器为一个总计数器。

    参数:
        counters: 计数器列表。

    返回:
        合并后的 Counter 对象。
    """
    total = Counter()
    for cnt in counters:
        total.update(cnt)
    return total


def save_counter_to_json(
    counter: Counter, output_path: str, tokenizer_path: str
) -> None:
    tokenizer = load_tokenizer(tokenizer_path)
    # 为所有唯一 token 生成“文本”表示（批量可加速，这里直接逐个处理）
    readable_result = []
    for token, count in counter.items():
        readable_text = tokenizer.convert_tokens_to_string([token])
        readable_text = readable_text.encode("utf-8", errors="replace").decode("utf-8")
        readable_result.append({"token": token, "text": readable_text, "count": count})

    # 按 token 排序（或按频次降序）
    readable_result.sort(key=lambda x: x["token"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(readable_result, f, ensure_ascii=False, indent=2)


def main() -> None:
    """程序入口。"""
    args = parse_arguments()

    # 1. 加载所有文本
    print("加载 Parquet 文本...")
    file_list = load_parquet_file_list(args.parquet_dir)
    print(f"共加载 {len(file_list)} 个文件。")

    # 2. 确定并行进程数
    num_workers = args.num_workers or cpu_count()
    print(f"使用 {num_workers} 个进程进行分词。")

    # 3. 将文本划分为多个 chunk，每个进程处理一个 chunk
    chunk_size = max(1, len(file_list) // num_workers)
    chunks = []
    for i in range(0, len(file_list), chunk_size):
        chunks.append(file_list[i : i + chunk_size])

    # 4. 多进程分词与局部统计
    # 将 tokenizer 路径传递给每个子进程，避免序列化 tokenizer 对象
    task_args = [(chunk, args.tokenizer_dir, args.text_column) for chunk in chunks]

    with Pool(processes=num_workers) as pool:
        partial_counters = pool.map(tokenize_chunk, task_args)

    # 5. 合并所有计数
    total_counter = merge_counters(partial_counters)
    print(f"不同 token 总数: {len(total_counter)}。")

    # 6. 输出 JSON
    save_counter_to_json(total_counter, args.output, args.tokenizer_dir)
    print(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
