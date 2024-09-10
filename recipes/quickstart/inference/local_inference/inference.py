import os
import sys
import time
import numpy as np
import pandas as pd
import gc
import fire

import json
import gradio as gr

import torch

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer
from llama_recipes.utils.evaluate import Evaluation

def main(
    model_name = "/scratch/qmei_root/qmei/xziyang/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1d54af340dc8906a2d21146191a9c184c35e47bd",
    my_prompt = "/scratch/qmei_root/qmei/xziyang/model_ckpt/llama3.1-70b-query-and-ref",
    peft_model: str = None,
    quantization: str = None, # Options: 4bit, 8bit
    max_new_tokens=100,  # The maximum numbers of tokens to generate
    prompt_file: str = "dataset_test.json",
    seed: int = 42,  # seed value for reproducibility
    do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
    top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
):
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        print("load peft")
        model = load_peft_model(model, peft_model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def inference(
        batch,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
    ):
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            print("hello!!")
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Model output:\n{output_text}")
        return output_text

    preds = []
    labels = []
    inputs = []
    evaluate = Evaluation(metric_names=["bleu","bertscore", "rouge", "meteor", "cosine","cosine","llmeval"])
    if prompt_file is not None:
        df = pd.read_json(prompt_file, lines=True)
        sampled_df = df[:200]
        example_column = sampled_df['example'].tolist()
        target_column = sampled_df['target'].tolist()
        for prompt,target in zip(example_column,target_column):
            lens = len(prompt)
            inputs.append(prompt)
            batch = tokenizer(
                prompt,
                truncation=True,
                max_length=max_padding_length,
                return_tensors="pt",
            )
            # print(f"batch:{batch}")
            # print(f"key:{batch.keys()}")
            output_text = inference(batch, temperature, top_p, top_k, max_new_tokens)
            output = output_text[lens:]
            preds.append(output)
            labels.append(target)
            torch.cuda.empty_cache()
            gc.collect()
            rst = {
                'input':prompt,
                'generate_predictions': output,
                'labels': target,
            }
            with open('pre.json', 'a') as f:
                json_line = json.dumps(rst)
                f.write(json_line + "\n")
                
        predictions_df = pd.DataFrame({'input':inputs, 'generate_predictions': preds, 'labels': labels})
        predictions_df.to_csv('pre.csv', index=False)
        # predictions_df.to_json('predictions_3.json', orient='records', lines=True)
        
        metrics = evaluate.compute(preds, labels)
        print(f"metrics:{metrics}")
        mean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, list):
                mean_metrics[k] = np.mean(v)  # 对列表计算均值
            elif isinstance(v, (int, float)):
                mean_metrics[k] = v  # 直接使用数值
            else:
                print(f"Skipping non-numeric value for key: {k}")
        print(f"mean_metrics:{mean_metrics}")
        with open('met.json', 'w') as json_file:
            json.dump(mean_metrics, json_file, indent=4)
        torch.cuda.empty_cache()
        gc.collect()
    
if __name__ == "__main__":
    model_name = "/scratch/qmei_root/qmei/xziyang/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1d54af340dc8906a2d21146191a9c184c35e47bd"
    # peft_model = "/scratch/qmei_root/qmei/xziyang/model_ckpt/llama3.1-70b-query-and-ref-2"
    fire.Fire(main(model_name=model_name,prompt_file="dataset_test.json"))
