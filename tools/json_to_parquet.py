#!/usr/bin/env python3
"""
流式处理 JSON 文件夹，按文件数量分片并保存为 Parquet 文件。

用法：
    python split_json_to_parquet.py --input_dir ./json_data --output_dir ./chunks --chunk_size 100
"""

import os
import glob
import argparse
import json
from pathlib import Path

import pandas as pd


def collect_json_files(folder_path: str) -> list:
    """
    获取指定文件夹下所有 .json 文件的路径列表。

    Args:
        folder_path: 目标文件夹路径。

    Returns:
        .json 文件路径列表（按名称排序）。
    """
    pattern = os.path.join(folder_path, "*.json")
    files = glob.glob(pattern)
    files.sort()
    return files


def save_chunk_as_parquet(messages_list: list, output_path: str) -> None:
    """
    将多个文件的 messages 列表保存为 Parquet 文件。

    Parquet 文件包含一列 'messages'，每行存储一个文件的 messages 列表。
    列表中的每个元素为包含 role、content、reasoning 的字典。

    Args:
        messages_list: 列表的列表，每个内层列表对应一个 JSON 文件的 messages 字段。
        output_path: 输出文件路径（应包含 .parquet 后缀）。
    """
    df = pd.DataFrame({"messages": messages_list})
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="gzip")


def stream_split_and_save(
    input_dir: str,
    output_dir: str,
    chunk_size_files: int = 100,
) -> None:
    """
    流式处理文件夹中的所有 .json 文件，按文件数量分片并保存为 Parquet。

    分片规则：
        - 每个分片包含固定数量的文件（chunk_size_files）。
        - 每个文件独立解析，提取其 "messages" 字段（若缺失则为空列表）。
        - 每次处理一个文件，内存中仅保留当前分片的数据。
        - 遇到无法解析的 JSON 文件会跳过并输出警告。

    Args:
        input_dir: 输入文件夹路径（包含 .json 文件）。
        output_dir: 输出文件夹路径（保存 .parquet 分片文件）。
        chunk_size_files: 每个分片包含的文件数量（默认 100）。
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_files = collect_json_files(input_dir)
    if not json_files:
        print(f"在 {input_dir} 中未找到任何 .json 文件。")
        return

    current_messages = []
    file_count = 0
    chunk_index = 0

    for file_path in json_files:
        print(f"处理文件: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  跳过文件 {file_path}：JSON 解析错误 - {e}")
            continue

        # 提取 messages 字段，不存在时使用空列表
        messages = data.get("messages", [])
        current_messages.append(messages)
        file_count += 1

        if file_count >= chunk_size_files:
            out_file = os.path.join(output_dir, f"chunk_{chunk_index:06d}.parquet")
            save_chunk_as_parquet(current_messages, out_file)
            print(f"  保存分片 {chunk_index:06d} ({file_count} 个文件)")
            chunk_index += 1
            current_messages = []
            file_count = 0

    # 保存最后一个分片（如果有剩余）
    if current_messages:
        out_file = os.path.join(output_dir, f"chunk_{chunk_index:06d}.parquet")
        save_chunk_as_parquet(current_messages, out_file)
        print(f"  保存最终分片 {chunk_index:06d} ({file_count} 个文件)")


def main():
    parser = argparse.ArgumentParser(
        description="将文件夹中的所有 .json 文件按文件数量分片并保存为 Parquet 格式"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="包含 .json 文件的输入文件夹路径",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="保存 .parquet 分片文件的输出文件夹路径",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="每个分片包含的文件数量（默认 100）",
    )
    args = parser.parse_args()

    stream_split_and_save(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size_files=args.chunk_size,
    )


if __name__ == "__main__":
    main()