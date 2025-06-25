import os
from huggingface_hub import delete_repo

# Replace with your dataset repo ID (e.g., "username/dataset_name")
delete_repo(repo_id="Kamel0/nsabb-mini-eval2",
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"))
