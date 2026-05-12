def save_lora(lora, tokenizer, lora_path):

    lora.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)