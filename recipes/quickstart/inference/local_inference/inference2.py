import os
import sys
import time
import numpy as np
import pandas as pd
import gc

import json
import gradio as gr

import torch

from accelerate.utils import is_xpu_available
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.inference.safety_utils import AgentType, get_safety_checker
from transformers import AutoTokenizer
from llama_recipes.utils.evaluate import Evaluation
import yaml
from llama_recipes.utils.dataset_utils import create_dataset,tokenize_llama_dataset
import random


def main(
    model_name,
    my_prompt: str = None,
    peft_model: str = None,
    config_file: str = None,
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
    if config_file:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

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

    raw_data = pd.read_csv(
        config["data"]["path"],
        sep="\t",
        usecols=[
            config["data"]["target_col"],
            config["data"]["time_col"],
            config["data"]["time_col"],
        ],
    )
    raw_data = raw_data.dropna(subset=[config["data"]["target_col"]])
    targets = raw_data[config["data"]["target_col"]]
    times = raw_data[config["data"]["time_col"]]

    npz = np.load(config["embeddings"]["path"])
    low_dim_embeddings = npz["low_dim_embeddings"]

    ds = create_dataset(
        texts=targets.to_list(),
        times=times,
        low_dim_embeddings=low_dim_embeddings,
        time_train=config["data"]["time_train"],
        time_val=config["data"]["time_val"],
        time_test=config["data"]["time_test"],
        use_sampler=config["data"]["use_sampler"],
        sampler_kwargs=config["data"]["sampler_kwargs"],
        input_kwargs=config["data"]["input_kwargs"],
    )
    ds = tokenize_llama_dataset(ds, tokenizer)
    dataset_test = ds["test"]
    # test_dataloader = torch.utils.data.DataLoader(
    #     dataset_test,
    #     num_workers=1,
    #     pin_memory=True,
    # )
    # dataloader_list = list(test_dataloader)
    # random_subset = random.sample(dataloader_list, min(1, len(dataloader_list)))
    # preds = []
    # labels = []
    # for batch in random_subset:
    #     batch = {key: value for key, value in batch.items() if key != "lens" }
    #     new_batch = {
    #         "input_ids": batch["examples"],
    #         "attention_mask": batch["examples_mask"]
    #     }
    #     print(f"new_batch:{new_batch}")
    #     labels = batch["labels"].to(batch["input_ids"].device)[0]
    #     output = inference(batch, temperature, top_p, top_k, max_new_tokens)
    #     target = tokenizer.decode(labels, skip_special_tokens=True)
    #     preds.append(output)
    #     labels.append(target)
    #     torch.cuda.empty_cache()
    #     gc.collect()
    # predictions_df = pd.DataFrame({'generate_predictions': preds, 'labels': labels})
    # predictions_df.to_json('pred.json', orient='records', lines=True)
    dataset_length = len(dataset_test)
    random_index = torch.randint(0, dataset_length, (1,)).item()
    random_example = torch.tensor(dataset_test[random_index]["examples"])
    random_example_mask = torch.tensor(dataset_test[random_index]["examples_mask"])
    random_target = dataset_test[random_index]["labels"]
    preds = []
    labels = []
    batch = {
        "input_ids": random_example.unsqueeze(0), 
        "attention_mask": random_example_mask.unsqueeze(0),  # 增加一个批次维度
    }
    print(f"batch:{batch}")
    lens = len(random_example)
    output = inference(batch, temperature, top_p, top_k, max_new_tokens)
    target = tokenizer.decode(random_target, skip_special_tokens=True)
    preds.append(output)
    labels.append(target)
    torch.cuda.empty_cache()
    gc.collect()
    predictions_df = pd.DataFrame({'generate_predictions': preds, 'labels': labels})
    predictions_df.to_json('pred.json', orient='records', lines=True)
    

if __name__ == "__main__":
    config_file = sys.argv[1]
    model_name = "/scratch/qmei_root/qmei/xziyang/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1d54af340dc8906a2d21146191a9c184c35e47bd"
    peft_model = "/scratch/qmei_root/qmei/xziyang/model_ckpt/llama3.1-70b-query-and-ref"
    main(model_name=model_name,peft_model=peft_model,config_file=config_file)
