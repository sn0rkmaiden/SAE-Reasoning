import os
import requests
import random
import re
from typing import List, Dict
import time
from collections import Counter
import datasets


QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {choice1}
B) {choice2}
C) {choice3}
D) {choice4}
""".strip()

# QUERY_TEMPLATE = "{Question}\n\nA) {choice1}\nB) {choice2}\nC) {choice3}\nD) {choice4}"
QUERY_TEMPLATE_API = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE = r"""
Look at the following question and an attempt by a student and extract which choice among A, B, C, D the student picked. If the student did not pick any choice, respond with "-1".

Examples:

    Question: ...
    Attempt: Answer: **A**

A

    Question: A) Dinosaur B) Elephant C) Cat D) Dog
    Attempt: ...The answer is therefore Elephant...

B

    Question: ...
    Attempt: Answer: None of the above

-1

    Question: ...
    Attempt: ...Answer: D), because...

D

    Question: ...
(A) 7 
(B) 8 
(C) 4 
(D) 10
    Attempt: 4

C

    Question: ...
    Attempt: ...\\boxed{C}...

C

---

YOUR TASK


Respond only with the capitalized alphabetic letter (without quotes) or -1. Do not include a rationale.

    Question: %(expression1)s
    Attempt: %(expression2)s
""".strip()


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)


def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])


class ChatCompletionSampler:
    """
    Sample from OpenRouter's chat completion API
    """

    _endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    _api_key: str = os.getenv("OPENROUTER_API_KEY", None)

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024
    ):
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}
    
    def __call__(self, message_list) -> str:
        if self._api_key is None:
            raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY environment variable.")

        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            payload = {
                "model": self.model, 
                "messages": message_list, 
                "temperature": self.temperature, 
                "max_tokens": self.max_tokens
            }
            response = requests.post(self._endpoint, headers={"Authorization": f"Bearer {self._api_key}"}, json=payload)
            if response.ok:
                return response.json()["choices"][0]["message"]["content"]
            else:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    response.status_code,
                )
                time.sleep(exception_backoff)
                trial += 1


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer(sampler: ChatCompletionSampler, question: str, attempt: str):
   prompt = EXTRACTION_TEMPLATE % {"expression1": question, "expression2": attempt}
   response = sampler([dict(content=prompt, role="user")])
   return response


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }
    
    processor = os.getenv("PROCESSOR", None)
    if processor is not None:
        sampler = ChatCompletionSampler(model=processor)
        question = QUERY_TEMPLATE_API.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
    else:
        print("Warning: no processor specified for GPQA evaluation.")
        sampler = None
    
    for i, a in enumerate(results, start=1):
        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match

        if a in ["a", "b", "c", "d"]:
            a = a.upper()

        if a not in ["A", "B", "C", "D"]:
            if sampler is not None:
                a = extract_answer(sampler, question, a)
            else:
                pass # TODO: Maybe add back legacy processing     
                
        if a not in ["A", "B", "C", "D"]:
            print(f"Warning: Default to A as given {results[i-1]} extracted {a}")
            a = "None"

        metrics["extracted_answers"].append(a)
        a = int(a == doc["answer"])
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + doc["answer"])
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(doc["answer"] == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval
