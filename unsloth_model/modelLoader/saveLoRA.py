def save_lora(lora, tokenizer, lora_path: str = "./lora"):

    # Save as safetensors format
    lora.save_pretrained(lora_path + "safetensors/")
    tokenizer.save_pretrained(lora_path + "safetensors/")
    # Save as GGUF format
    lora.save_pretrained_gguf(lora_path + "gguf/", tokenizer, quantization_method = "q8_0")