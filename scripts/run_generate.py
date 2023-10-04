import argparse
import time

import torch
from transformers import AutoTokenizer, set_seed

from optimum.neuron import NeuronModelForCausalLM


# python run_generate.py --model NousResearch/Llama-2-7b-chat-hf --batch_size 2 --num_cores 2 --auto_cast_type bf16 --save_dir /home/ubuntu/llama-2-7b-chat-hf --prompts "Count from 0 to 100 and then sum the values step by step." --sequence_length 2048 --temperature 1.0 --seed 42


def generate(model, tokenizer, prompts, length, temperature):
    # Specifiy padding options for decoder-only architecture
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Encode tokens and generate using temperature
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    start = time.time()
    with torch.inference_mode():
        sample_output = model.generate(
            **tokens, do_sample=True, max_length=length, temperature=temperature, top_k=50, top_p=0.9
        )
    end = time.time()
    outputs = [tokenizer.decode(tok) for tok in sample_output]
    return outputs, (end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-chat-hf", type=str, help="The HF Hub model id or a local directory.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="The batch size.",
    )
    parser.add_argument(
        "--num_cores", type=int, default=2, help="The number of cores on which the model should be split."
    )
    parser.add_argument(
        "--auto_cast_type", type=str, default="bf16", choices=["f32", "f16", "bf16"], help="One of f32, f16, bf16."
    )
    parser.add_argument(
        "--save_dir", type=str, help="The save directory. Allows to avoid recompiling the model every time."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="Count from 0 to 100 and then sum the values step by step.",
        help="The prompts to use for generation, using | as separator.",
    )
    parser.add_argument("--sequence_length", type=int, default=2048, help="The number of tokens in the generated sequences.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature to generate. 1.0 has no effect, lower tend toward greedy sampling.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Pass a seed for reproducibility.")
    args = parser.parse_args()
    print(args)
    
    start_time = time.time()
    model = NeuronModelForCausalLM.from_pretrained(
        args.model,
        export=True,
        low_cpu_mem_usage=True,
        # These are parameters required for the conversion
        batch_size=args.batch_size,
        num_cores=args.num_cores,
        auto_cast_type=args.auto_cast_type,
        n_positions=args.sequence_length
        
    )
    print(f"Compilation took {time.time() - start_time} seconds")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # if args.seed is not None:
    #     set_seed(args.seed)
    # model = NeuronModelForCausalLM.from_pretrained(args.model, export=False, low_cpu_mem_usage=True)
    prompt = "Write a super long story about a llama?"
    prompts = [prompt] * args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    outputs, latency = generate(model, tokenizer, prompts, args.sequence_length, args.temperature)
    print(outputs)
    print(f"{len(outputs)} outputs generated using Neuron model in {latency:.4f} s")
        
        
        