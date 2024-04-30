import argparse
import time
import json

import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def time_eval(model_name, repeats=5, prompt=None, promptlen=100, genlen=100, temperature=1.0, topk=1, topp=1.0, minp=0.0, repetition_penalty=1.0, batch=1):
    device = "cuda"
    dtype = torch.float16

    print(f"Loading model {model_name}")
    is_mamba = model_name.startswith("state-spaces/mamba-")
    if is_mamba:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device}, torch_dtype=dtype)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    torch.random.manual_seed(0)
    if prompt is None:
        input_ids = torch.randint(1, 1000, (batch, promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + genlen

    if is_mamba:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            min_p=minp,
            repetition_penalty=repetition_penalty,
        )
    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=max_length,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_k=topk,
            top_p=topp,
            repetition_penalty=repetition_penalty,
        )
    out = fn()
    if prompt is not None:
        print(tokenizer.batch_decode(out.sequences.tolist()))
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")

    time_list = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.time()
        fn()
        torch.cuda.synchronize()
        end = time.time()
        time_list.append((end-start)*1000)
    return time_list

# 100 Token Generation Time for Given Prompt
print("Model: Mamba 130M Given Prompt")
time_list = time_eval("state-spaces/mamba-130m", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 160M Given Prompt")
time_list = time_eval("EleutherAI/pythia-160m", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 370M Given Prompt")
time_list = time_eval("state-spaces/mamba-370m", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 410M Given Prompt")
time_list = time_eval("EleutherAI/pythia-410m", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 790M Given Prompt")
time_list = time_eval("state-spaces/mamba-790m", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 1B Given Prompt")
time_list = time_eval("EleutherAI/pythia-1b", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 1.4B Given Prompt")
time_list = time_eval("state-spaces/mamba-1.4b", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 1.4B Given Prompt")
time_list = time_eval("EleutherAI/pythia-1.4b", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 2.8B Given Prompt")
time_list = time_eval("state-spaces/mamba-2.8b", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 2.8B Given Prompt")
time_list = time_eval("EleutherAI/pythia-2.8b", prompt="My cat wrote all this CUDA code for a new language model and")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")


# 100 Token Generation Time for Random Prompt
print("Model: Mamba 130M Random Prompt")
time_list = time_eval("state-spaces/mamba-130m")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 160M Random Prompt")
time_list = time_eval("EleutherAI/pythia-160m")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 370M Random Prompt")
time_list = time_eval("state-spaces/mamba-370m")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 410M Random Prompt")
time_list = time_eval("EleutherAI/pythia-410m")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 790M Random Prompt")
time_list = time_eval("state-spaces/mamba-790m")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 1B Random Prompt")
time_list = time_eval("EleutherAI/pythia-1b")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 1.4B Random Prompt")
time_list = time_eval("state-spaces/mamba-1.4b")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 1.4B Random Prompt")
time_list = time_eval("EleutherAI/pythia-1.4b")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Mamba 2.8B Random Prompt")
time_list = time_eval("state-spaces/mamba-2.8b")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")

print("Model: Pythia 2.8B Random Prompt")
time_list = time_eval("EleutherAI/pythia-2.8b")
print("Avg Time: "+str(np.mean(time_list)))
print("Variance: "+str(np.std(time_list)))
print("\n")