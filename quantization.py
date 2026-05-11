import torch
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8lbit=True,
)

model_path = "./qwen_model/qwen3.5/0.8B_base_f16/"
save_path = "./qwen_model/qwen3.5/0.8B_base_8bit/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", quantization_config=bnb_config
)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
