import json
import string
from collections import Counter
import re
import numpy as np


def acc_choice(predictions, answers):
    num_correct = 0
    total = len(predictions)

    for pred, ans_list in zip(predictions, answers):
        pred = pred.lower()

        correct = False
        for ans in ans_list:
            ans = ans.lower()
            if re.search(rf"\b{ans}\b", pred):
                correct = True
                break
        if correct:
            num_correct += 1

    acc = round(100 * num_correct / total, 2)
    return acc


def acc_score(predictions, answers):
    num_correct = 0
    for id, answer in enumerate(answers):
        pred = predictions[id]
        correctness = (
            "True" if any(ans.lower() in pred.lower() for ans in answer) else "False"
        )
        if correctness == "True":
            num_correct += 1
        else:
            pass
    acc = num_correct / len(answers)
    return round(100 * acc, 2)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def F1_scorer(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(score, qa_f1_score(prediction, ground_truth))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def compute_exact(predictions, answers):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        for ground_truth in ground_truths:
            score = max(
                score,
                int(normalize_answer(prediction) == normalize_answer(ground_truth)),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False



