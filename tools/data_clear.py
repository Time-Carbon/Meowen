#!/usr/bin/env python3
import os
import argparse
import logging
import multiprocessing
from typing import List, Set

import pandas as pd
import ahocorasick

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(processName)s: %(message)s"
)
logger = logging.getLogger(__name__)

# 全局变量，用于子进程共享自动机
_global_automaton = None


def load_sensitive_words(csv_path: str) -> List[str]:
    """
    从 CSV 文件加载敏感词列表。
    要求 CSV 包含列名 `text`，每行一个敏感词。
    返回小写化的敏感词列表（去重）。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"敏感词文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError("CSV 文件缺少 'text' 列，请检查列名。")

    # 去除空值、转为小写、去重
    words = (
        df["text"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    logger.info(f"已加载 {len(words)} 个敏感词。")
    return words


def build_automaton(words: List[str]) -> ahocorasick.Automaton:
    """
    根据敏感词列表构建 Aho-Corasick 自动机（不区分大小写）。
    自动机中存储小写形式的敏感词。
    """
    automaton = ahocorasick.Automaton()
    for idx, word in enumerate(words):
        automaton.add_word(word, word)  # value 存储原词（小写）
    automaton.make_automaton()
    logger.info("Aho-Corasick 自动机构建完成。")
    return automaton


def count_distinct_matches(text: str, automaton: ahocorasick.Automaton) -> int:
    """
    返回文本 `text` 中匹配到的不同敏感词数量。
    文本会被转为小写再进行匹配。
    """
    text_lower = text.lower()
    matched: Set[str] = set()
    for end_idx, word in automaton.iter(text_lower):
        matched.add(word)
        # 如果已经达到2个可提前终止（仅用于行级判断时由外部控制，这里不作提前退出）
    return len(matched)


def row_contains_at_least_five_keywords(
    row: pd.Series, automaton: ahocorasick.Automaton
) -> bool:
    """
    判断 DataFrame 的一行中，所有字符串列内容包含的不同敏感词数量是否 >= 2。
    是则返回 True（需要删除），否则 False。
    """
    matched_set: Set[str] = set()
    for value in row:
        if isinstance(value, str):
            text_lower = value.lower()
            for _, word in automaton.iter(text_lower):
                matched_set.add(word)
                if len(matched_set) >= 5:
                    return True
    return len(matched_set) >= 5


def init_worker(automaton: ahocorasick.Automaton):
    """进程池初始化函数，将自动机设置为全局变量。"""
    global _global_automaton
    _global_automaton = automaton


def process_single_file(args: tuple) -> str:
    """
    处理单个 Parquet 分片文件。
    输入：(input_path, output_path)
    使用全局自动机 `_global_automaton` 进行敏感词匹配。
    返回处理完毕的文件路径。
    """
    input_path, output_path = args
    automaton = _global_automaton
    if automaton is None:
        raise RuntimeError("自动机未初始化，请检查进程池初始化函数。")

    logger.info(f"开始处理: {input_path}")

    # 读取分片
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        logger.error(f"读取文件 {input_path} 失败: {e}")
        raise

    if df.empty:
        # 空文件直接另存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"文件为空，直接保存: {output_path}")
        return input_path

    # 筛选出至少包含 2 个不同敏感词的行（标记删除）
    mask_to_delete = df.apply(
        lambda row: row_contains_at_least_five_keywords(row, automaton),
        axis=1
    )

    # 保留不满足删除条件的行
    cleaned_df = df[~mask_to_delete]
    deleted_count = mask_to_delete.sum()
    logger.info(
        f"文件 {os.path.basename(input_path)} 完成: "
        f"总行数 {len(df)}, 删除 {deleted_count} 行, 保留 {len(cleaned_df)} 行。"
    )

    # 保存清洗后的分片，保持原始列和类型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_parquet(output_path, index=False)

    return input_path


def main():
    parser = argparse.ArgumentParser(
        description="高效敏感词过滤：删除 Parquet 分片中匹配 >=2 个不同敏感词的行"
    )
    parser.add_argument(
        "--sensitive-csv", required=True, help="敏感词 CSV 文件路径（列名: text）"
    )
    parser.add_argument(
        "--input-dir", required=True, help="输入 Parquet 分片所在文件夹"
    )
    parser.add_argument(
        "--output-dir", required=True, help="输出清洗后 Parquet 分片的文件夹"
    )
    parser.add_argument(
        "--pattern", default="*.parquet", help="匹配分片文件的通配符，默认 *.parquet"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="并行进程数，默认为 CPU 核心数",
    )
    args = parser.parse_args()

    # 1. 加载敏感词并构建自动机（主进程）
    words = load_sensitive_words(args.sensitive_csv)
    automaton = build_automaton(words)

    # 2. 收集所有待处理的分片文件路径
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"输入目录不存在: {args.input_dir}")
    import glob
    file_paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not file_paths:
        logger.warning("未找到任何匹配的 Parquet 文件，退出。")
        return
    logger.info(f"共发现 {len(file_paths)} 个分片文件。")

    # 3. 准备输入输出路径对
    tasks = []
    for fp in file_paths:
        rel_path = os.path.relpath(fp, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        tasks.append((fp, out_path))

    # 4. 多进程并行处理
    logger.info(f"启动 {args.num_workers} 个工作进程...")
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(automaton,),
    ) as pool:
        # 使用 imap_unordered 可实时看到进度，并且内存友好
        for finished_path in pool.imap_unordered(process_single_file, tasks):
            logger.info(f"完成: {finished_path}")

    logger.info("全部文件处理完毕。")


if __name__ == "__main__":
    main()