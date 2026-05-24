import argparse
import logging
import sys
from pathlib import Path

import torch
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="使用bitsandbytes对模型进行8bit量化"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="原始模型的路径或HuggingFace模型ID"
    )
    
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="量化后模型保存的目录路径"
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="设备映射配置，默认为auto"
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="是否信任远程代码"
    )
    
    return parser.parse_args()


def create_quantization_config() -> BitsAndBytesConfig:
    """
    创建8bit量化配置
    
    Returns:
        BitsAndBytesConfig: 量化配置对象
    """
    logger.info("创建8bit量化配置...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    return bnb_config


def load_tokenizer(model_path: str, trust_remote_code: bool = False) -> AutoTokenizer:
    """
    加载tokenizer
    
    Args:
        model_path: 模型路径
        trust_remote_code: 是否信任远程代码
        
    Returns:
        AutoTokenizer: 加载的tokenizer对象
    """
    logger.info(f"正在加载tokenizer: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code
    )
    
    logger.info("Tokenizer加载成功")
    return tokenizer


def load_and_quantize_model(
    model_path: str,
    quantization_config: BitsAndBytesConfig,
    device_map: str = "auto",
    trust_remote_code: bool = False
) -> AutoModelForCausalLM:
    """
    加载模型并进行8bit量化
    
    Args:
        model_path: 模型路径
        quantization_config: 量化配置
        device_map: 设备映射配置
        trust_remote_code: 是否信任远程代码
        
    Returns:
        AutoModelForCausalLM: 量化后的模型对象
    """
    logger.info(f"正在加载并量化模型: {model_path}")
    logger.info(f"设备映射: {device_map}")
    logger.info(f"量化配置: 8bit")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code
    )
    
    logger.info("模型加载和量化完成")
    return model


def save_model_and_tokenizer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str
) -> None:
    """
    保存量化后的模型和tokenizer
    
    Args:
        model: 模型对象
        tokenizer: tokenizer对象
        save_path: 保存路径
    """
    logger.info(f"正在保存模型和tokenizer到: {save_path}")
    
    # 创建保存目录
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(save_path)
    logger.info("模型保存成功")
    
    # 保存tokenizer
    tokenizer.save_pretrained(save_path)
    logger.info("Tokenizer保存成功")
    
    logger.info(f"所有文件已保存到: {save_path}")


def main() -> None:
    """
    主函数：执行模型量化和保存的完整流程
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        logger.info("=" * 60)
        logger.info("开始模型8bit量化流程")
        logger.info("=" * 60)
        
        # 创建量化配置
        quantization_config = create_quantization_config()
        
        # 加载tokenizer
        tokenizer = load_tokenizer(
            args.model_path,
            trust_remote_code=args.trust_remote_code
        )
        
        # 加载并量化模型
        model = load_and_quantize_model(
            args.model_path,
            quantization_config,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code
        )
        
        # 保存模型和tokenizer
        save_model_and_tokenizer(model, tokenizer, args.save_path)
        
        logger.info("=" * 60)
        logger.info("模型8bit量化完成！")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()