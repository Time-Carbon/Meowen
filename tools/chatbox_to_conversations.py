import json
import argparse
import os
import sys


def get_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def load_json(file_path):
    with open(file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def find_schema(data, target):
    """递归搜索 target 键，返回从根到该键的路径列表，找不到返回空列表"""
    def search(current, path):
        if isinstance(current, dict):
            for key, value in current.items():
                if key == target:
                    return path + [key]
                result = search(value, path + [key])
                if result:
                    return result
        elif isinstance(current, list):
            for idx, item in enumerate(current):
                result = search(item, path + [idx])
                if result:
                    return result
        return None

    result = search(data, [])
    return result if result is not None else []


def convert_to_conversations(messages):
    conversations = []
    for frame in messages:
        # 安全提取文本内容
        try:
            text = frame["contentParts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # 如果缺失则跳过或设为空字符串，根据需要决定
            continue

        conv = {
            "role": frame.get("role", "unknown"),
            "content": text,
        }
        conversations.append(conv)
    return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 chatbox 的 json 记录转换为 conversations 数据集"
    )
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    file_path = get_path(args.file)
    jsonData = load_json(file_path)

    # 找到 messages 的路径
    target_schema = find_schema(jsonData, "messages")
    if not target_schema:
        print("错误：在输入文件中未找到 'messages' 字段，请确认文件格式。", file=sys.stderr)
        sys.exit(1)

    # 沿着路径深入定位到 messages 列表
    messages = jsonData
    for key in target_schema:
        messages = messages[key]

    # 转换为对话格式
    conversations = convert_to_conversations(messages)

    output_file_path = get_path(args.output)
    with open(output_file_path, mode="w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共 {len(conversations)} 条对话，已保存至 {output_file_path}")