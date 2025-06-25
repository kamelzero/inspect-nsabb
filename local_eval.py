from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import f1, Score, scorer
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai import eval_set
from multilabel_f1 import multilabel_f1
import json
import os

with open('task_data.json') as fp:
    task_data = json.load(fp)

@task
def task_nsabb_mini_eval(ind=0):
    return Task(
        dataset=MemoryDataset([
            Sample(
                input=task_data[ind]["input"],
                choices=task_data[ind]["choices"],
                target=task_data[ind]['target']
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

os.system("rm -rf ./logs-local")
success, logs = eval_set(
   [task_nsabb_mini_eval(ind=ind) for ind in range(len(task_data))],
   model=["google/gemini-1.5-pro"],
   log_dir="logs-local"
)
