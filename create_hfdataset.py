import json
import argparse
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo, delete_repo

def prepare_dataset(task_data, split="validation"):
    # Create dataset and wrap it in DatasetDict the split name
    dataset = Dataset.from_list(task_data)
    dataset_dict = DatasetDict({split: dataset})
    return dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_data_path", type=str, default="data/task_data.json")
    parser.add_argument("--dataset_name", type=str, default="nsabb-mini-eval")
    parser.add_argument("--split", type=str, default='validation')
    parser.add_argument("--private", type=bool, default=False)
    args = parser.parse_args()

    with open(args.task_data_path, "r") as f:
        task_data = json.load(f)
    dataset = prepare_dataset(task_data)

    try:
        delete_repo(args.dataset_name, repo_type="dataset")
    except:
        pass
    create_repo(args.dataset_name, repo_type="dataset", private=args.private)
    dataset.push_to_hub(args.dataset_name)

    print(f"Dataset {args.dataset_name} ({'private' if args.private else 'public'}) with {len(task_data)} tasks pushed to Hugging Face Hub")
