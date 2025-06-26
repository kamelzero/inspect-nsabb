import json

def strip_answer(text):
    answer = text.strip().replace("ANSWER:", "").strip()
    return answer

def normalize_answer(text: str):
    return set(s for s in text.upper() if s in {"A", "B", "C", "D", "E", "F", "G", "H"})

def parse_log_sample(log_samples):
    task_target_answer_lst = []
    for task_index, sample in enumerate(log_samples):
        target = sample['target']
        choices = sample['output']['choices']
        if len(choices) == 0:
            continue
        msg = choices[0]['message']
        model = msg['model']
        answer = msg['content']
        if type(answer) == list:
            answer = [a for a in answer if 'text' in a]
            answer = answer[0]['text'] if answer else None
        task_target_answer_lst.append(dict(task_index=task_index, model=model, answer=answer, target=target))
    return task_target_answer_lst

def structured_eval_samples_io(log_samples):
    lst = parse_log_sample(log_samples)
    for el in lst:
        el['answer'] = list(normalize_answer(strip_answer(el['answer'])))
        el['target'] = list(el['target'])
    return lst
