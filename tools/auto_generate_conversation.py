#!/usr/bin/env python3
"""
Multi-agent conversation generator (v3).
Key improvements:
1. Full context preservation: opener is always visible to Agent A.
2. Perspective switching: each agent sees history from its own role
   (self -> assistant, partner -> user), preventing content confusion.
3. Clean output: final conversation excludes system prompts and opener,
   with Agent A as "user" and Agent B as "assistant".
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel, Field


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate coherent multi-turn conversations between two LLM agents."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.openai.com/v1",
        help="Base URL for API requests (default: OpenAI)",
    )
    parser.add_argument(
        "--temperature", default=1.0, type=float, help="Temperature (default=1.0)"
    )
    parser.add_argument(
        "--frequency_penalty",
        default=1.0,
        type=float,
        help="Frequency penalty (default=1.0)",
    )
    parser.add_argument(
        "--top_p", default=0.95, type=float, help="Top P (default=0.95)"
    )
    parser.add_argument(
        "--min_p", default=0.05, type=float, help="Min P (default=0.05)"
    )
    parser.add_argument("--top_k", default=20, type=int, help="Top K (default=20)")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument(
        "--num-sessions",
        type=int,
        required=True,
        help="Number of independent conversations to generate",
    )
    parser.add_argument(
        "--prompt-a", required=True, help="Path to system prompt for Agent A (.txt)"
    )
    parser.add_argument(
        "--prompt-b", required=True, help="Path to system prompt for Agent B (.txt)"
    )
    parser.add_argument(
        "--opener", required=True, help="Path to CSV file containing opening lines"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the generated conversations",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of conversation turns (default: 10)",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row of the CSV file (e.g., a header)",
    )
    return parser.parse_args()


def load_text_file(filepath: str) -> str:
    """Load the entire content of a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise SystemExit(f"Error loading {filepath}: {e}")


def load_openers(filepath: str, skip_header: bool = False) -> List[Dict[str, str]]:
    """Load opening lines from a two‑column CSV (think, respond)."""
    openers = []
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            if skip_header:
                try:
                    next(reader)
                except StopIteration:
                    raise SystemExit("CSV file is empty (no data after header).")
            for row in reader:
                # 至少需要两列，且两列非空
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    openers.append({"think": row[0].strip(), "respond": row[1].strip()})
    except Exception as e:
        raise SystemExit(f"Error loading openers from {filepath}: {e}")
    if not openers:
        raise SystemExit("No opening lines found in the CSV file.")
    return openers


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_retries: int = 3,
    temperature: float = 1.0,
    frequency_penalty: float = 1.0,
    top_p: float = 0.95,
    min_p: float = 0.05,
    top_k: float = 20,
    response_format: Dict = None,  # 新增：支持 JSON 模式
) -> str:
    """Send a chat completion request and return the response text."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "extra_body": {
                    "enable_thinking": False,
                    "top_k": top_k,
                    "min_p": min_p,
                },
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(
                f"API call failed (attempt {attempt+1}/{max_retries}): {e}",
                file=sys.stderr,
            )
            last_exception = e
    raise RuntimeError(
        f"Chat completion failed after {max_retries} attempts"
    ) from last_exception


def generate_conversation(
    client: OpenAI,
    model: str,
    prompt_a: str,
    prompt_b: str,
    opener: Dict,
    turns: int = 10,
    top_p: float = 0.95,
    min_p: float = 0.05,
    top_k: float = 20,
    temperature: float = 1.0,
    frequency_penalty: float = 1.0,
) -> List[Dict]:
    """
    Generate a multi-turn conversation using perspective switching.

    Agent A's perspective:
        system: prompt_a
        user: opener
        (assistant: A1, user: B1, assistant: A2, user: B2, ...)
    Agent B's perspective:
        system: prompt_b
        user: A1
        (assistant: B1, user: A2, assistant: B2, ...)

    Final output is cleaned to contain only the exchange in the format:
        [user: A1, assistant: B1, user: A2, assistant: B2, ...]
    """

    # 启用 JSON 模式
    class json_format_schema(BaseModel):
        reasoning: str = Field(
            description="用于输出角色的内心独白",
            strict=True,
        )
        responding: str = Field(
            description="用于输出角色的外在表达",
            strict=True,
        )

    json_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "chatbot",
            "schema": json_format_schema.model_json_schema(),
        },
    }

    # Convert the opener to a JSON string for initializing Agent A's history.
    opener_a = json.dumps(
        {"query": opener["respond"]},
        ensure_ascii=False,
    )

    # Message histories for each agent, including system prompt and mapped roles
    messages_a = [
        {"role": "system", "content": prompt_a},
        {"role": "user", "content": opener_a},
    ]

    # Convert the opener to a JSON string for initializing Agent B's history.
    opener_b = json_format_schema.model_dump_json(
        json_format_schema(reasoning=opener["think"], responding=opener["respond"]),
        ensure_ascii=False,
    )
    messages_b = [
        {
            "role": "system",
            "content": f"{prompt_b}",
        },
        {"role": "assistant", "content": opener_b},
    ]

    # Final output
    output_messages = [
        {
            "role": "system",
            "reasoning": "",
            "content": prompt_a,
        },
        {
            "role": "user",
            "reasoning": "",
            "content": opener["respond"],
        },
    ]

    # 按照目标轮数生成对话
    for i in range(turns):
        # Agent A generates a response (it sees itself as assistant)
        response_a = chat_completion(
            client,
            model,
            messages_a,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            response_format=json_format,
        )
        response_a = json_format_schema.model_validate_json(response_a)

        # Update A's history: its own reply is assistant
        messages_a.append(
            {
                "role": "assistant",
                "content": json_format_schema.model_dump_json(
                    self=response_a, ensure_ascii=False
                ),
            }
        )
        # Update B's history: A's reply is user
        messages_b.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"query": response_a.responding}, ensure_ascii=False
                ),
            }
        )

        # Record in output as user
        output_messages.append(
            {
                "role": "assistant",
                "reasoning": response_a.reasoning,
                "content": response_a.responding,
            }
        )

        # Agent B generates a response (it sees itself as assistant)
        if i < (turns - 1):
            response_b = chat_completion(
                client,
                model,
                messages_b,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                response_format=json_format,
            )
            response_b = json_format_schema.model_validate_json(response_b)

            # Update B's history: its own reply is assistant
            messages_b.append(
                {
                    "role": "assistant",
                    "content": json_format_schema.model_dump_json(
                        self=response_b, ensure_ascii=False
                    ),
                }
            )
            # Update A's history: B's reply is user
            messages_a.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": response_b.responding}, ensure_ascii=False
                    ),
                }
            )

            # Record in output as assistant
            output_messages.append(
                {
                    "role": "user",
                    "reasoning": "",
                    "content": response_b.responding,
                }
            )

    return output_messages


def save_conversation(
    conversation: List[Dict], output_dir: str, session_index: int
) -> None:
    """Save a conversation to a JSON file."""
    filename = f"session_{session_index:04d}.json"
    filepath = Path(output_dir) / filename
    conversation = {"messages": conversation}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    print(f"Saved: {filepath}")


def main() -> None:
    try:
        args = parse_arguments()

        if not args.api_key:
            sys.exit(
                "API key not provided. Use --api-key or set OPENAI_API_KEY environment variable."
            )

        prompt_a = load_text_file(args.prompt_a)
        prompt_b = load_text_file(args.prompt_b)
        openers = load_openers(args.opener, skip_header=args.skip_header)

        os.makedirs(args.output_dir, exist_ok=True)

        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

        for session_idx in range(args.num_sessions):
            opener = openers[session_idx % len(openers)]
            print(
                f"Generating session {session_idx+1}/{args.num_sessions} "
                f"with opener: {opener["respond"][:50]}...",
                flush=True,
            )

            conversation = generate_conversation(
                client=client,
                model=args.model,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                opener=opener,
                turns=args.turns,
                temperature=args.temperature,
                frequency_penalty=args.frequency_penalty,
                min_p=args.min_p,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            save_conversation(conversation, args.output_dir, session_idx + 1)

        print(f"All {args.num_sessions} sessions generated successfully.")

    except KeyboardInterrupt:
        sys.exit("\nAborted by user.")
    except Exception as e:
        sys.exit(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
