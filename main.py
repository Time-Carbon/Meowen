from unsloth import FastLanguageModel
import torch

max_context = 4096
model_path = "./model/"

def import_model(max_context, path):
    
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = path,
            max_seq_length = max_context,
            dtype = None,
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
            gpu_memory_utilization = 0.8
            )
    
    return model, tokenizer

def create_lora(model, rank, lora_list):

    lora = FastLanguageModel.get_peft_model(
            model = model,
            r = rank,
            target_modules = lora_list,
            lora_alpha = 2 * rank,
            lora_dropout=0.0,
            bias='none',
            use_gradient_checkpointing='unsloth'
            )

    return lora


if __name__ == "__main__":

    model, tokenizer = import_model(max_context, model_path)

    lora_list = [
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj'
            ]
    create_lora(model, 8, lora_list)
