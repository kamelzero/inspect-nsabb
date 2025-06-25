from inspect_ai.scorer import Score, scorer
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Score, Target, accuracy, scorer

@scorer(metrics=[accuracy()])
def multilabel_f1():
    async def score(state: TaskState, target: Target):
        def normalize(text: str):
            return set(s for s in text.upper() if s in {"A", "B", "C", "D", "E", "F", "G", "H"})

        model_output = state.output.completion.strip().replace("ANSWER:", "").strip()
        target_text = target.text.strip()

        model_set = normalize(model_output)
        target_set = normalize(target_text)

        print(f"Model output: {model_output} | Target: {target_text}")
        print(f"Model output normed: {model_set} | Target normed: {target_set}")

        # Simple F1 calculation (can replace with better later)
        tp = len(model_set & target_set)
        fp = len(model_set - target_set)
        fn = len(target_set - model_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return Score(value=f1)

    return score
