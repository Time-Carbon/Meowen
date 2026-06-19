import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Iterator, Tuple
import argparse


def get_total_rows_and_file_info(folder: str) -> Tuple[int, List[dict]]:
    """
    扫描文件夹下所有 .parquet 文件，返回总行数以及每个文件的元信息。
    元信息列表：每个元素为 dict，包含 'path', 'num_rows', 'row_groups'。
    row_groups: 列表，每个元素为 (起始全局索引, 行组内行数)
    """
    parquet_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"在文件夹 {folder} 中未找到 .parquet 文件")

    file_infos = []
    global_offset = 0
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        num_rows = pf.metadata.num_rows
        # 获取每个行组的行数
        rg_offsets = []
        for rg in range(pf.metadata.num_row_groups):
            rg_metadata = pf.metadata.row_group(rg)
            rg_num_rows = rg_metadata.num_rows
            rg_offsets.append((global_offset, rg_num_rows))
            global_offset += rg_num_rows
        file_infos.append(
            {"path": path, "num_rows": num_rows, "row_groups": rg_offsets}
        )
    return global_offset, file_infos


def generate_permutation_on_disk(n: int, seed: int = 42) -> np.memmap:
    """在磁盘上生成一个随机排列，并返回 memmap 对象。"""
    # 创建临时文件存放排列，使用 np.memmap
    perm_file = "permutation.npy"
    perm = np.memmap(perm_file, dtype=np.int64, mode="w+", shape=(n,))
    perm[:] = np.arange(n, dtype=np.int64)
    # 用指定随机种子打乱
    rng = np.random.RandomState(seed)
    rng.shuffle(perm)  # 原地打乱
    return perm


def iter_shuffled_rows(
    perm: np.memmap, file_infos: List[dict], batch_size: int = 10000
) -> Iterator[pd.Series]:
    """
    按随机排列顺序逐批生成行（pandas Series 对象），每次生成 batch_size 行。
    内部会批量读取原始文件，减少 I/O 次数。
    """
    total = len(perm)
    # 构建索引到 (file_idx, rg_idx, offset_in_rg) 的快速查找
    # 因为需要多次二分查找，构建一个列表 (start_global_index, file_idx, rg_idx)
    index_map = []
    for fi, info in enumerate(file_infos):
        for rg_idx, (start, rg_rows) in enumerate(info["row_groups"]):
            index_map.append((start, fi, rg_idx, rg_rows))
    # 按 start 排序以支持二分
    index_map.sort(key=lambda x: x[0])
    starts = [x[0] for x in index_map]

    def locate_global_index(global_idx: int):
        """根据全局索引返回 (file_idx, rg_idx, offset_in_rg)"""
        # 二分查找最后一个 start <= global_idx
        pos = np.searchsorted(starts, global_idx, side="right") - 1
        start, fi, rg_idx, rg_rows = index_map[pos]
        offset = global_idx - start
        return fi, rg_idx, offset

    for batch_start in range(0, total, batch_size):
        batch_indices = perm[
            batch_start : batch_start + batch_size
        ]  # 这是排列中的一段，已经乱序
        # 按文件分组，批量读取
        file_group = {}  # file_idx -> list of (rg_idx, offset)
        # 先收集所有需要的行
        for global_idx in batch_indices:
            fi, rg_idx, offset = locate_global_index(global_idx)
            file_group.setdefault(fi, []).append((rg_idx, offset, global_idx))

        # 按文件处理
        rows_with_idx = []  # 存储 (global_idx, text)
        for fi, items in file_group.items():
            path = file_infos[fi]["path"]
            pf = pq.ParquetFile(path)
            # 按行组分组，一次读取一个行组中需要的所有行
            rg_items = {}
            for rg_idx, offset, gidx in items:
                rg_items.setdefault(rg_idx, []).append((offset, gidx))
            for rg_idx, offsets in rg_items.items():
                # 读取整个行组（只读 text 列）
                table = pf.read_row_group(rg_idx, columns=["text"])
                df_rg = table.to_pandas()
                # 提取需要的行
                for offset, gidx in offsets:
                    text = df_rg.iloc[offset]["text"]
                    rows_with_idx.append((gidx, text))
        # 按 global_idx 排序，恢复批次内顺序
        rows_with_idx.sort(key=lambda x: x[0])
        for _, text in rows_with_idx:
            yield text


def split_by_char_count_stream(
    row_iterator: Iterator[str], char_limit: int = 10_000_000
) -> List[pd.DataFrame]:
    """与原始 split_by_char_count 逻辑相同，但输入为迭代器。"""
    shards = []
    buffer_rows = []
    buffer_chars = 0
    for text in row_iterator:
        text_len = len(text)
        if text_len >= char_limit:
            if buffer_rows:
                shards.append(pd.DataFrame(buffer_rows, columns=["text"]))
                buffer_rows = []
                buffer_chars = 0
            shards.append(pd.DataFrame([{"text": text}], columns=["text"]))
            continue
        if buffer_chars + text_len > char_limit:
            shards.append(pd.DataFrame(buffer_rows, columns=["text"]))
            buffer_rows = []
            buffer_chars = 0
        buffer_rows.append({"text": text})
        buffer_chars += text_len
    if buffer_rows:
        shards.append(pd.DataFrame(buffer_rows, columns=["text"]))
    return shards


def save_shards(
    shards: List[pd.DataFrame], output_dir: str, prefix: str = "part"
) -> None:
    """
    将分片保存为 Parquet 文件，命名格式为 {prefix}_{index:04d}.parquet。
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, shard in enumerate(shards, start=1):
        filename = f"{prefix}_{idx:04d}.parquet"
        filepath = os.path.join(output_dir, filename)
        shard.to_parquet(filepath, index=False, compression="gzip")
        print(f"已保存: {filepath} (行数: {len(shard)})")


def process_parquet_stream(
    input_dir: str,
    output_dir: str,
    char_limit: int = 10_000_000,
    shuffle: bool = False,
    batch_size: int = 10000,
) -> None:
    """
    流式处理完整流程。
    """
    print("正在扫描文件并统计行数...")
    total_rows, file_infos = get_total_rows_and_file_info(input_dir)
    print(f"总行数: {total_rows}")

    if shuffle:
        print("正在生成随机排列（存储于磁盘）...")
        perm = generate_permutation_on_disk(total_rows, seed=42)
        print("开始按随机顺序流式读取...")
        row_iter = iter_shuffled_rows(perm, file_infos, batch_size)
        # 使用完后删除排列文件（可选）
        # os.remove('permutation.npy')
    else:
        # 不打乱时，按文件顺序逐行读取（流式）
        def iter_sequential():
            for info in file_infos:
                pf = pq.ParquetFile(info["path"])
                for rg in range(pf.metadata.num_row_groups):
                    table = pf.read_row_group(rg, columns=["text"])
                    df = table.to_pandas()
                    for _, row in df.iterrows():
                        yield row["text"]

        row_iter = iter_sequential()

    print(f"正在按每 {char_limit} 字符进行分片...")
    shards = split_by_char_count_stream(row_iter, char_limit)
    print(f"分片完成，共产生 {len(shards)} 个分片。")

    print("正在保存分片...")
    save_shards(shards, output_dir)  # 复用原 save_shards 函数
    print("全部完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按字符数分片 parquet 文本数据")
    parser.add_argument("input_dir", help="包含 .parquet 文件的输入文件夹")
    parser.add_argument("output_dir", help="输出分片的目标文件夹")
    parser.add_argument("--shuffle", action="store_true", help="进行数据打乱")
    parser.add_argument(
        "--char-limit",
        type=int,
        default=10_000_000,
        help="每个分片的字符数上限（默认 10,000,000）",
    )
    args = parser.parse_args()

    process_parquet_stream(args.input_dir, args.output_dir, args.char_limit, args.shuffle)
