#!/usr/bin/env python3
"""
流式处理文本文件夹，按字符数分片并保存为 Parquet 文件。

用法：
    python split_txt_to_parquet.py --input_dir ./txt_data --output_dir ./chunks
"""

import os
import glob
import argparse
from pathlib import Path

import pandas as pd


def collect_txt_files(folder_path: str) -> list:
    """
    获取指定文件夹下所有 .txt 文件的路径列表。

    Args:
        folder_path: 目标文件夹路径。

    Returns:
        .txt 文件路径列表（按名称排序）。
    """
    pattern = os.path.join(folder_path, "*.txt")
    files = glob.glob(pattern)
    files.sort()  # 确保处理顺序一致
    return files


def save_chunk_as_parquet(text: list, output_path: str) -> None:
    """
    将单个文本块保存为 Parquet 文件。

    Parquet 文件包含一列 'text'，每行存储一个块。

    Args:
        text: 待保存的文本内容。
        output_path: 输出文件路径（应包含 .parquet 后缀）。
    """
    df = pd.DataFrame({"text": text})
    df.to_parquet(output_path, index=False, engine="pyarrow")


def stream_split_and_save(
    input_dir: str,
    output_dir: str,
    chunk_size_chars: int = 1_000_000,
) -> None:
    """
    流式处理文件夹中的所有 .txt 文件，按字符数分片并保存为 Parquet。

    分片规则：
        - 每个分片尽量接近 chunk_size_chars 字符。
        - 不截断任何文本块：当加入新块会使分片超出限制时，将当前分片保存，
          然后新块作为下一个分片的开头（跨文件边界同理）。
        - 采用流式读取，一次仅将一个文件的部分内容载入内存。

    Args:
        input_dir: 输入文件夹路径（包含 .txt 文件）。
        output_dir: 输出文件夹路径（保存 .parquet 分片文件）。
        chunk_size_chars: 每个分片的目标字符数（默认 100 万）。
        read_buffer: 每次从文件中读取的字符数（默认 10000）。
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    txt_files = collect_txt_files(input_dir)
    if not txt_files:
        print(f"在 {input_dir} 中未找到任何 .txt 文件。")
        return

    current_chunk = []
    current_len = 0
    chunk_index = 0

    for file_path in txt_files:
        print(f"处理文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            while True:
                block = f.read()
                if not block:
                    break

                # 若当前块能放入当前分片，则直接追加
                if current_len + len(block) <= chunk_size_chars:
                    current_chunk.append(block)
                    current_len += len(block)
                else:
                    # 保存当前分片（如果不为空）
                    if current_chunk:
                        out_file = os.path.join(
                            output_dir, f"chunk_{chunk_index:06d}.parquet"
                        )
                        save_chunk_as_parquet(current_chunk, out_file)
                        print(
                            f"  保存分片 {chunk_index:06d} ({len(current_chunk)} 字符)"
                        )
                        chunk_index += 1

                    # 当前块作为新分片的开头（不截断）
                    current_chunk = [block]
                    current_len = len(block)

    # 保存最后一个分片
    if current_chunk:
        out_file = os.path.join(output_dir, f"chunk_{chunk_index:06d}.parquet")
        save_chunk_as_parquet(current_chunk, out_file)
        print(f"  保存最终分片 {chunk_index:06d} ({current_len} 字符)")


def main():
    parser = argparse.ArgumentParser(
        description="将文件夹中的所有 .txt 文件按字符数分片并保存为 Parquet 格式"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="包含 .txt 文件的输入文件夹路径",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="保存 .parquet 分片文件的输出文件夹路径",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1_000_000,
        help="每个分片的目标字符数（默认 1000000）",
    )
    args = parser.parse_args()

    stream_split_and_save(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size_chars=args.chunk_size,
    )


if __name__ == "__main__":
    main()
