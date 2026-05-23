import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


def parse_args() -> Path:
    """解析命令行参数，返回日志文件路径。"""
    if len(sys.argv) != 2:
        print("Usage: python extract_logs.py <log_file>")
        sys.exit(1)
    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    return file_path


def read_log_lines(file_path: Path) -> List[str]:
    """读取日志文件的所有行。"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.readlines()


def extract_log_entries(lines: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    从日志行中提取训练和评估数据字典。
    返回两个列表：train_entries 和 eval_entries。
    """
    train_entries = []
    eval_entries = []

    # 匹配类似 {'key': 'value', ...} 的字典字符串
    dict_pattern = re.compile(r"\{.*?\}")

    for line in lines:
        match = dict_pattern.search(line)
        if not match:
            continue

        dict_str = match.group(0)
        try:
            data = ast.literal_eval(dict_str)
        except (ValueError, SyntaxError):
            continue

        if not isinstance(data, dict):
            continue

        if 'eval_loss' in data:
            # 评估日志：只保留 loss（映射自 eval_loss）和 epoch
            try:
                entry = {
                    'loss': float(data['eval_loss']),
                    'epoch': float(data['epoch'])
                }
                eval_entries.append(entry)
            except (KeyError, ValueError):
                # 忽略字段缺失或转换失败的行
                continue
        elif 'loss' in data and 'eval_loss' not in data:
            # 训练日志：保留 loss, grad_norm, learning_rate, epoch
            try:
                entry = {
                    'loss': float(data['loss']),
                    'grad_norm': float(data['grad_norm']),
                    'learning_rate': float(data['learning_rate']),
                    'epoch': float(data['epoch'])
                }
                train_entries.append(entry)
            except (KeyError, ValueError):
                continue

    return train_entries, eval_entries


def save_to_excel(entries: List[Dict[str, Any]], columns: List[str], output_path: Path):
    """将字典列表保存为 Excel 文件。"""
    if not entries:
        print(f"Warning: No entries to save for '{output_path.name}'. Creating empty file.")
        # 创建空 DataFrame 仍保留列名
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(entries, columns=columns)
    df.to_excel(output_path, index=False)
    print(f"Saved {len(entries)} rows to '{output_path}'.")


def main():
    log_path = parse_args()
    lines = read_log_lines(log_path)
    train_entries, eval_entries = extract_log_entries(lines)

    # 定义输出列顺序，确保与示例一致
    train_columns = ['loss', 'grad_norm', 'learning_rate', 'epoch']
    eval_columns = ['loss', 'epoch']

    save_to_excel(train_entries, train_columns, Path('train.xlsx'))
    save_to_excel(eval_entries, eval_columns, Path('eval.xlsx'))


if __name__ == '__main__':
    main()