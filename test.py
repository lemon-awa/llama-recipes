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


if __name__ == "__main__":
    evaluate = Evaluation(metric_names=["bleu","bertscore", "rouge", "meteor", "cosine","cosine","llmeval"])
    data = []
    with open("predictions_2.json", 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    preds = [item["generate_predictions"] for item in data]
    labels = [item["labels"] for item in data]
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
    # mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    print(f"mean_metrics:{mean_metrics}")
    with open('mean_metrics_2.json', 'w') as json_file:
        json.dump(mean_metrics, json_file, indent=4)