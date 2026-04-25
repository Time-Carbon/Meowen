### Unsloth related
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch

### Dataset related
from datasets import load_dataset

### Train related
from trl import SFTTrainer
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainingArguments

### Global data
max_context = 4096
model_path = "./model/"
lora_path = "./lora/"

### Main process
def import_model():
    
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_context,
            dtype = None,
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
            gpu_memory_utilization = 0.8,
            fast_inference = False
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

def SFTtrain(lora, tokenizer, dataset, steps, lr, regularization):

    trainer = SFTTrainer(
            model = lora,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_context,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 0.1 * steps,
                max_steps = steps,
                learning_rate = lr,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 0.1 * steps,
                optim = "adamw_8bit",
                weight_decay = regularization,
                lr_scheduler_type = "linear"
                )
            )

    return lora

def save_lora(lora, tokenizer):

    lora.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

if __name__ == "__main__":

    model, tokenizer = import_model()

    lora_list = [
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj'
            ]
    lora = create_lora(model, 8, lora_list)
