from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import f1, Score, scorer
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai import eval_set
from multilabel_f1 import multilabel_f1
from inspect_ai.dataset import hf_dataset
import json
import os

# with open('task_data.json') as fp:
#     task_data = json.load(fp)

# @task
# def task_nsabb_mini_eval(ind=0):
#     return Task(
#         dataset=MemoryDataset([
#             Sample(
#                 input=task_data[ind]["input"],
#                 choices=task_data[ind]["choices"],
#                 target=task_data[ind]['target']
#             ),
#         ]),
#         solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
#         scorer=multilabel_f1()
#     )

@task
def task_nsabb_mini_eval_hf():
    return Task(
        dataset=hf_dataset(
            path="Kamel0/nsabb-mini-eval",
            split="validation",
        ),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

os.system("rm -rf ./logs-local")
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

from inspect_ai.log import list_eval_logs, read_eval_log_samples
results = list_eval_logs("logs-local")
eval_filename = results[0].name
logsamples_iter = read_eval_log_samples(eval_filename)
log_samples = [sample.model_dump() for sample in logsamples_iter]
with open("logs-local/eval_samples.json", "w") as fp:
    json.dump(log_samples, fp, indent=4)
