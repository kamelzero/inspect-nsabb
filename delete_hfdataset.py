"""
This script is used to delete the Hugging Face dataset
specified in hf_config.json.
"""

import os
from huggingface_hub import delete_repo
import json

with open("hf_config.json", "r") as f:
    hf_config = json.load(f)

# Replace with your dataset repo ID (e.g., "username/dataset_name")
delete_repo(repo_id=hf_config["hf_dataset"],
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"))
