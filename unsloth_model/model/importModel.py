from unsloth import FastLanguageModel


def import_model(model_path, mem_usage, load_to_vllm = False, max_RL_context = 4096):

    qwen_template = """
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' -}}
    {%- elif message['role'] == 'user' -%}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n<think>\n' -}}
{%- endif -%}
"""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_RL_context,
        dtype=None,
        load_in_4bit=False,
        load_in_8bit=True,
        use_gradient_checkpointing="unsloth",
        gpu_memory_utilization=mem_usage,
        fast_inference=load_to_vllm,
    )

    tokenizer.chat_template = qwen_template.strip()

    return model, tokenizer
