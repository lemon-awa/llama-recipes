import sys
import yaml
import pandas as pd
from llama_recipes.utils.dataset_utils import create_new_dataset
import numpy as np
import json

def main(config_file):
    if config_file:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    
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

    ds = create_new_dataset(
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

    dataset_test = ds["test"]
    dataset_test.to_json("dataset_test.json", orient='records', lines=True)


if __name__ == "__main__":
    config_file = sys.argv[1]
    main(config_file=config_file)
