import numpy as np
import yaml
import torch
from datasets import load_dataset

from datasets import Dataset

# dev_dataset = Dataset.from_list(dev_samples)
# print(dev_dataset)

import json


def check_splits(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    splits = list(data.keys())
    print(f"Available splits: {splits}")

    # Print sample size for each split
    for split in splits:
        if isinstance(data[split], list):
            print(f"{split} split size: {len(data[split])} examples")


# Usage


if __name__=='__main__':
    pathname = 'data/eval/hellaswag.json'

    # If your JSON file is named "data.json", and you want to treat the "dev" array as your dataset:
    dataset = load_dataset(
        'json',
        data_files={'dev': pathname},
        field='dev'  # <-- This tells load_dataset to look inside "dev" for the records
    )

    print(dataset)
    # check_splits(pathname)
    # exit()
    # dataset = load_dataset('json', data_files=pathname)
    #
    # dataset = load_dataset("json", data_files=pathname, field=['dev', 'test'])
    # # Load dataset from local path
    # # dataset = load_dataset('hellaswag',
    # #                        data_dir=pathname,  # Replace with your dataset path
    # #                        )
    #
    # print(dataset)





