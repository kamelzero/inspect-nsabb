from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import f1, Score, scorer
from inspect_ai.solver import multiple_choice, system_message
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.scorer import Score, Target, accuracy, scorer

@task
def multi_nsabb():
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