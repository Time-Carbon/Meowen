import os
import pandas as pd
from typing import List


def load_parquet_files(folder: str) -> pd.DataFrame:
    """
    自动加载指定文件夹下所有 .parquet 文件，合并为一个 DataFrame。
    要求每个文件必须包含 'text' 列。
    """
    parquet_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"在文件夹 {folder} 中未找到 .parquet 文件")

    frames = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        if "text" not in df.columns:
            raise ValueError(f"文件 {file_path} 缺少必需的 'text' 列")
        frames.append(df[["text"]])  # 只保留 text 列
    combined = pd.concat(frames, ignore_index=True)
    return combined


def shuffle_dataframe(df: pd.DataFrame, shuffle: bool) -> pd.DataFrame:
    """随机打乱 DataFrame 的行顺序。"""
    raw_size = len(df)
    df = df.drop_duplicates()
    duplicated_size = len(df)
    print(f"去重{raw_size - duplicated_size}行")
    if shuffle:
        print("正在打乱数据...")
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        return df


def split_by_char_count(
    df: pd.DataFrame, char_limit: int = 10_000_000
) -> List[pd.DataFrame]:
    """
    按字符数对 DataFrame 进行分片，保证每一行的文本完整不被切割。
    若某一行本身的字符数 >= char_limit，则单独作为一个分片。
    """
    shards = []
    buffer_rows = []
    buffer_chars = 0

    for _, row in df.iterrows():
        text = row["text"]
        text_len = len(text)

        # 处理超长行：直接单独成为一个分片
        if text_len >= char_limit:
            # 先保存当前缓冲区
            if buffer_rows:
                shards.append(pd.DataFrame(buffer_rows, columns=["text"]))
                buffer_rows = []
                buffer_chars = 0
            shards.append(pd.DataFrame([row], columns=["text"]))
            continue

        # 如果加入当前行会超出限制，则保存当前缓冲区并开始新的缓冲区
        if buffer_chars + text_len > char_limit:
            shards.append(pd.DataFrame(buffer_rows, columns=["text"]))
            buffer_rows = []
            buffer_chars = 0

        # 将当前行加入缓冲区
        buffer_rows.append(row)
        buffer_chars += text_len

    # 保存剩余的缓冲区
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


def process_parquet(
    input_dir: str, output_dir: str, char_limit: int = 10_000_000, shuffle: bool = False
) -> None:
    """
    完整处理流程：
    1. 加载所有 parquet 文件
    2. 合并并打乱
    3. 按字符数分片
    4. 保存分片
    """
    print("正在加载数据...")
    df = load_parquet_files(input_dir)
    print(f"加载完成，共 {len(df)} 行。")

    df = shuffle_dataframe(df, shuffle)

    print(f"正在按每 {char_limit} 字符进行分片...")
    shards = split_by_char_count(df, char_limit)
    print(f"分片完成，共产生 {len(shards)} 个分片。")

    print("正在保存分片...")
    save_shards(shards, output_dir)
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

    process_parquet(args.input_dir, args.output_dir, args.char_limit, args.shuffle)
