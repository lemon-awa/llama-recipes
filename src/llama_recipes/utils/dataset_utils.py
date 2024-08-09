# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

import torch

from llama_recipes.dataset import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_llamaguard_toxicchat_dataset,
)

from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from llama_recipes.utils.common import KNNSampler
from typing import Dict, Any
import datasets

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,

}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )


def make_input(
    query: np.ndarray,
    reference_embeddings: np.ndarray = None,
    reference_texts: List[str] = None,
    instruction: str = "Convert the coordinate to text",
    split: str = " | ",
) -> str:
    np.set_printoptions(precision=4)
    # Example:
    # Convert the coordinate to text: [1, 2] | [3, 4] reference_text_1 | [1, 4]
    # reference_text_2
    prompts = [f"{instruction}: {query}"]
    if reference_embeddings is not None:
        assert reference_texts is not None
        assert len(reference_embeddings) == len(reference_texts)
        for i in range(len(reference_embeddings)):
            prompts.append(f"{reference_embeddings[i]} {reference_texts[i]}")
    return split.join(prompts)


def create_dataset(
    texts: List[str],
    times: List[int],
    low_dim_embeddings: np.ndarray,
    time_train: Tuple[int, int],
    time_val: Tuple[int, int],
    time_test: Tuple[int, int],
    use_sampler: bool = False,
    sampler_kwargs: Dict[str, Any] = None,
    input_kwargs: Dict[str, Any] = None,
) -> datasets.DatasetDict:
    times = np.array(times)
    train_mask = (times >= time_train[0]) & (times < time_train[1])
    val_mask = (times >= time_val[0]) & (times < time_val[1])
    test_mask = (times >= time_test[0]) & (times < time_test[1])

    if use_sampler:
        sampler = KNNSampler(low_dim_embeddings, times, **sampler_kwargs)
    else:
        sampler = None

    if input_kwargs is None:
        input_kwargs = {}

    def get_data_split(mask):
        inputs = []
        targets = []
        examples = []
        for i in range(len(texts)):
            if mask[i]:
                if sampler is not None:
                    time = times[i]
                    indices, dists = sampler.sample(low_dim_embeddings[i], time)
                    reference_embeddings = low_dim_embeddings[indices, :]
                    reference_texts = [texts[j] for j in indices]
                    input = make_input(
                            low_dim_embeddings[i],
                            reference_embeddings=reference_embeddings,
                            reference_texts=reference_texts,
                            **input_kwargs,
                        )
                    inputs.append(f"{input} {texts[i]}")
                    examples.append(input)
                else:
                    input = make_input(low_dim_embeddings[i], **input_kwargs)
                    inputs.append(f"{input} {texts[i]}")
                    examples.append(input)
                targets.append(f"{input} {texts[i]}")

        return datasets.Dataset.from_dict({"text": inputs, "target": targets, "example": examples})

    train_dataset = get_data_split(train_mask)
    val_dataset = get_data_split(val_mask)
    test_dataset = get_data_split(test_mask)

    return datasets.DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )


def tokenize_llama_dataset(
    ds: datasets.DatasetDict, tokenizer: AutoTokenizer
) -> datasets.DatasetDict:
    ds = ds.map(
        lambda x: tokenizer(x["text"], truncation=False, padding="max_length", max_length=512),
        remove_columns=["text"],
        batched=True,
    ).map(
        lambda x: {
            "labels": tokenizer(
                x["target"], truncation=False, padding="max_length", max_length=512
            )["input_ids"]
        },
        batched=True,
        remove_columns=["target"],
    ).map(
        lambda x: {
            "examples": tokenizer(
                x["example"], truncation=False, padding="max_length", max_length=512
            )["input_ids"]
        },
        batched=True,
    ).map(
        lambda x: {
            "examples_mask": tokenizer(
                x["example"], truncation=False, padding="max_length", max_length=512
            )["attention_mask"]
        },
        batched=True,
    ).map(
        lambda x: {
            "lens": [len(tokenizer(
                ex, truncation=False, padding=False,
            )["input_ids"]) for ex in x["example"]]
        },
        batched=True,
        remove_columns=["example"],
    )
    return ds