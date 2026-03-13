import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from peft import PeftModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *

def set_random_seed(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(path):
    if path.endswith("json"):
        return json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        return data
    raise NotImplementedError()

def infer(args, test_data, answer_extraction_fn):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prompts = []

    for example in test_data:
        prompt = ""
        for mess in example['messages']:
            if mess['role'] == 'user':
                if args.model_type == 'llama3':
                    token_info = f"{tokenizer.eos_token}{args.compression_ratio}{tokenizer.eos_token}{tokenizer.eos_token}" if args.compression_ratio < 1.0 else f"{tokenizer.eos_token}"
                    prompt += f"{tokenizer.bos_token}<|start_header_id|>user<|end_header_id|>\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{mess['content']}\n{token_info}<|start_header_id|>assistant<|end_header_id|>\n\n"
                elif args.model_type == 'qwen':
                    if args.compression_ratio < 1.0:
                        prompt += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{mess['content']}<|eot_id|>{args.compression_ratio}<|eot_id|><|im_end|>\n<|im_start|>assistant\n"
                    else:
                        prompt += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{mess['content']}<|im_end|>\n<|im_start|>assistant\n"
            elif mess['role'] == 'assistant':
                prompt += mess['content'].rstrip()
        example['prompt'] = prompt.lstrip()
        prompts.append(example['prompt'])

    print(" >> Loading model...")
    model_path_to_load = args.model_path

    if args.use_adapter:
        merged_path = os.path.join(args.adapter_path, "merged_static_weights")
        if not os.path.exists(merged_path):
            base_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
            model_peft = PeftModel.from_pretrained(base_model, args.adapter_path)
            merged_model = model_peft.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            del base_model, model_peft, merged_model
            torch.cuda.empty_cache()
        model_path_to_load = merged_path

    if args.use_vllm:
        model = LLM(
            model=model_path_to_load, 
            tokenizer=model_path_to_load, 
            trust_remote_code=True, 
            tensor_parallel_size=1, 
            max_model_len=4096, 
            device="cuda"
        )
        
        torch.cuda.synchronize()
        # TIMESTAMP PER BASH (WARM START)
        try:
            with open("timing_info.json", "w") as f:
                json.dump({"start_inference": int(time())}, f)
        except Exception as e:
            print(f"WARNING: Timing sync failed: {e}")

        start_time = time()
        outputs = model.generate(prompts, SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_new_tokens, stop=[tokenizer.eos_token]))
        torch.cuda.synchronize()
        total_time = time() - start_time
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs = [output.outputs[0].text for output in outputs]
    else:
        # Fallback Transformers
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        start_time = time()
        outputs, _ = generate_completions(model=model, tokenizer=tokenizer, prompts=prompts, max_new_tokens=args.max_new_tokens, batch_size=args.eval_batch_size)
        total_time = time() - start_time

    # Post-processing
    results = []
    for example, output in zip(test_data, outputs):
        cot = output.split('\n\nThe final answer is:')[0]
        cot_len = tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
        pred = eval(answer_extraction_fn)(example['messages'][-2]['content'], output, task='cot')
        item = deepcopy(example)
        item.update({'model_output': output, 'prediction': pred, 'cot_length': cot_len})
        results.append(item)
    
    return results, total_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--model-size", type=str, default="7b")
    parser.add_argument("--model-type", type=str, default="qwen")
    parser.add_argument("--use_adapter", action='store_true')
    parser.add_argument("--compression_ratio", type=float, default=1.0)
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--data-type", type=str, default="test")
    parser.add_argument("--use_vllm", action='store_true')
    parser.add_argument("--max_num_examples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    test_conf = read_data(f"configs/{args.benchmark}_{args.data_type}.json")

    for src, info in test_conf.items():
        raw_data = read_data(info['test_path'])
        test_data = []
        for i, sample in enumerate(raw_data):
            processed = eval(info['process_fn'])(sample)
            for j, item in enumerate(processed):
                item['id'] = f"{src}-test-{i}-{j}"
                item['reference'] = [m['content'] for m in item['messages'] if m['role'] == 'assistant']
                for m in item['messages']: 
                    if m['role'] == 'assistant': m['content'] = ''
                test_data.append(item)

        if len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        results, total_time = infer(args, test_data, info['answer_extraction_fn'])

        # Scoring & Save
        labels = [eval_math(item) for item in results]
        for item, label in zip(results, labels): item['accuracy'] = label
        
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            json.dump({
                "n_samples": len(results),
                "accuracy": sum(labels) / len(results),
                "total_inference_time": total_time
            }, fout, indent=4)