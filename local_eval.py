"""
This script is used to evaluate the model on the NSABB-mini-eval dataset.
Log files are saved in the logs-local directory.
"""

from inspect_ai import Task, task
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai import eval_set
from inspect_ai.dataset import hf_dataset
from inspect_ai.log import list_eval_logs, read_eval_log_samples
from multilabel_f1 import multilabel_f1
from log_utils import structured_eval_samples_io
import os
import pandas as pd
import questionary
import shutil
import json

with open("hf_config.json", "r") as f:
    hf_config = json.load(f)

@task
def task_nsabb_mini_eval_hf():
    return Task(
        dataset=hf_dataset(
            path=hf_config["hf_dataset"],
            split="validation",
        ),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

logs_dir = "./logs-local"

if os.path.exists(logs_dir):
    answer = questionary.confirm(f"Do you really want to delete '{logs_dir}'?").ask()
    if answer:
        shutil.rmtree(logs_dir)
        print("Deleted.")
    else:
        print("Aborted.")
        exit(1)

success, logs = eval_set(
   task_nsabb_mini_eval_hf(), #[task_nsabb_mini_eval(ind=ind) for ind in range(len(task_data))],
   model=[
          "google/gemini-1.5-pro",
          "google/gemini-2.0-flash",
          "google/gemini-2.5-flash-preview-05-20",
          "google/gemini-2.5-pro-preview-03-25",
          "openai/gpt-4o",
          "openai/o1",
          "anthropic/claude-3-7-sonnet-latest",
          "anthropic/claude-sonnet-4-20250514",
          "mistral/mistral-large-latest"
          ],
   log_dir="logs-local"
)

results = list_eval_logs("logs-local")
structured_samples_lst = []
for ind, result in enumerate(results):
    eval_filename = result.name
    logsamples_iter = read_eval_log_samples(eval_filename)
    log_samples = [sample.model_dump() for sample in logsamples_iter]
    structured_samples_lst.append(structured_eval_samples_io(log_samples))

csv_fn = f"{logs_dir}/structured_samples.csv"
pd.DataFrame(structured_samples_lst).to_csv(csv_fn, index=False)
print(f"Results saved to {csv_fn}")