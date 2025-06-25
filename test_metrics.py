"""
Test to show that a custom multilabel_f1 metric works as expected
for multiple choice tasks with multiple targets.

F1 = 2 × (precision × recall) / (precision + recall)
where:
    Precision = (number of correct predicted labels) / (total predicted labels)
    Recall = (number of correct predicted labels) / (total target labels)
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import f1, Score, scorer
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai import eval_set
import math
import pandas as pd

@task
def test_f1_singletarget():
    """
    Built-in f1 metric is used.
    The model will return 2 letters, but the target has only 1.

    Target: ["D"]
    Prediction: ["C", "D"]
 
    Correct predictions: ["D"]
    Precision: 1 correct / 2 predicted = 0.5
    Recall: 1 correct / 1 target = 1.0

    Expected F1=2×0.5×1.00.5+1.0=1.01.5≈0.6667
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=f1()
    )

@task
def test_f1_multitarget_correct():
    """
    Built-in f1 metric is used.
    The model will return 2 letters, and the target has 2.

    Target: ["C", "D"]
    Prediction: ["C", "D"]

    Correct predictions: ["C", "D"]
    Precision: 2 correct / 2 predicted = 1.0
    Recall: 2 correct / 2 target = 1.0

    Expected F1=2×1.0×1.01.0+1.0=1.0
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["C", "D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=f1()
    )

@task
def test_f1_multitarget_partiallycorrect():
    """
    Built-in f1 metric is used.
    The model will return 2 letters, and the target has 2,
    but target will deliberately only have 1 of the 2 answers the model will provide.

    Target: ["A", "D"]
    Prediction: ["C", "D"]

    Correct predictions: ["D"] (only 1 of the 2 target answers is correct)
    Precision: 1 correct / 2 predicted = 0.5
    Recall: 1 correct / 2 target = 0.5

    Expected F1=2×0.5×0.50.5+0.5=0.51.0=0.5
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["A", "D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=f1()
    )

@task
def test_multilabelf1_singletarget():
    """
    Custom multilabel_f1 metric is used.
    The model will return 2 letters, but and the target has only 1.

    Target: ["D"]
    Prediction: ["C", "D"]
 
    Correct predictions: ["D"]
    Precision: 1 correct / 2 predicted = 0.5
    Recall: 1 correct / 1 target = 1.0

    Expected F1=2×0.5×1.00.5+1.0=1.01.5≈0.6667
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

@task
def test_multilabelf1_multitarget_correct():
    """
    Custom multilabel_f1 metric is used.
    The model will return 2 letters, but and the target has 2.

    Target: ["C", "D"]
    Prediction: ["C", "D"]

    Correct predictions: ["C", "D"]
    Precision: 2 correct / 2 predicted = 1.0
    Recall: 2 correct / 2 target = 1.0

    Expected F1=2×1.0×1.01.0+1.0=1.0    
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["C", "D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

@task
def test_multilabelf1_multitarget_partiallycorrect():
    """
    Custom multilabel_f1 metric is used.
    The model will return 2 letters, but and the target has 2,
    but target will deliberately only have 1 of the 2 answers the model will provide.


    Target: ["A", "D"]
    Prediction: ["C", "D"]

    Correct predictions: ["D"] (only 1 of the 2 target answers is correct)
    Precision: 1 correct / 2 predicted = 0.5
    Recall: 1 correct / 2 target = 0.5

    Expected F1=2×0.5×0.50.5+0.5=0.51.0=0.5    
    """
    return Task(
        dataset=MemoryDataset([
            Sample(
                input="What is the square root of 25? Select all that apply.",
                choices=["A) 0", "B) 1", "C) -5", "D) 5"],
                target=["A", "D"]
            ),
        ]),
        solver=[system_message("Select all that apply. ANSWER:"), multiple_choice()],
        scorer=multilabel_f1()
    )

@scorer(metrics=[accuracy()])
def multilabel_f1():
    async def score(state: TaskState, target: Target):
        def normalize(text: str):
            return set(s for s in text.upper() if s in {"A", "B", "C", "D", "E", "F", "G", "H"})

        model_output = state.output.completion.strip().replace("ANSWER:", "").strip()
        target_text = target.text.strip()

        model_set = normalize(model_output)
        target_set = normalize(target_text)

        print(f"Model: {model_set} | Target: {target_set}")

        # Simple F1 calculation (can replace with better later)
        tp = len(model_set & target_set)
        fp = len(model_set - target_set)
        fn = len(target_set - model_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return Score(value=f1)

    return score

test_functions = [test_f1_singletarget(), 
                  test_f1_multitarget_correct(),
                  test_f1_multitarget_partiallycorrect(),
                  test_multilabelf1_singletarget(),
                  test_multilabelf1_multitarget_correct(),
                  test_multilabelf1_multitarget_partiallycorrect()]

expected_scores = {
    "test_f1_singletarget": 0.6667,
    "test_f1_multitarget_correct": 1.0,
    "test_f1_multitarget_partiallycorrect": 0.5,
    "test_multilabelf1_singletarget": 0.6667,
    "test_multilabelf1_multitarget_correct": 1.0,
    "test_multilabelf1_multitarget_partiallycorrect": 0.5}

success, logs = eval_set(
   test_functions,
   model=["google/gemini-1.5-pro"],
   log_dir="logs-test-metrics"
)

assert len(logs) == len(test_functions)
assert len(logs) == len(expected_scores)
results_by_score = []
for ind, log in enumerate(logs):
    task_name = log.model_dump().get('eval').get('task_registry_name')
    score = log.model_dump().get('results').get('scores')[0]
    score_name = score.get('name')
    if score.get('metrics').get('accuracy'):
        score_value = score.get('metrics').get('accuracy').get('value')
    else:
        score_value = score.get('metrics').get('mean').get('value')
    expected = expected_scores.get(task_name)
    results_by_score.append({
        'task_name': task_name,
        'score_name': score_name,
        'score_value': score_value,
        'expected': expected,
        'match': math.isclose(score_value, expected, rel_tol=1e-3, abs_tol=0.0)
    })

df = pd.DataFrame(results_by_score) 
print(df.groupby('score_name').match.mean())
