import torch
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
)

model_path = "./model/qwen3/1.7B_Base/"
save_path = "./model/qwen3/1.7B_Base_4bit"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", quantization_config=bnb_config
)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
