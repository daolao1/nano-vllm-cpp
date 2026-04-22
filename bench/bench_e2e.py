#!/usr/bin/env python3
"""Benchmark nano-vllm (Python) with 5 prompts for comparison with C++ version."""
import os, time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
PROMPTS = [
    "What is the capital of France?",
    "用Python写一个快速排序",
    "Explain quantum computing in simple terms",
    "1+1等于几？",
    "Write a haiku about programming",
]
MAX_TOKENS = 256

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=1e-9, max_tokens=MAX_TOKENS)

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # Encode to count prompt tokens.
    prompt_token_counts = [len(tokenizer.encode(f)) for f in formatted]
    total_prompt_tokens = sum(prompt_token_counts)

    t0 = time.perf_counter()
    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    total_gen_tokens = sum(len(o["token_ids"]) for o in outputs)
    print(f"\n{'='*60}")
    print(f"nano-vllm (Python) Benchmark")
    print(f"{'='*60}")
    print(f"Prompts:          {len(PROMPTS)}")
    print(f"Prompt tokens:    {total_prompt_tokens}")
    print(f"Generated tokens: {total_gen_tokens}")
    print(f"Total time:       {elapsed:.2f}s")
    print(f"Prefill:          {total_prompt_tokens/elapsed:.0f} tok/s")
    print(f"Decode:           {total_gen_tokens/elapsed:.0f} tok/s")
    print(f"{'='*60}")

    for prompt, output in zip(PROMPTS, outputs):
        text = output["text"]
        # Truncate for display.
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"\nQ: {prompt}")
        print(f"A: {text}")

if __name__ == "__main__":
    main()
