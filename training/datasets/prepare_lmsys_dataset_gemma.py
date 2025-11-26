import os
import random
import math
import fire

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from transformer_lens.utils import tokenize_and_concatenate

class Reasoner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prompt(self, input):
        """Convert LMSYS conversation to Gemma chat format"""
        conversation = input["conversation"]
        chat_history = []

        for turn in conversation:
            role = turn["role"]
            content = turn["content"]

            if role == "system":
                chat_history.append({"role": "system", "content": content})
            elif role == "user":
                chat_history.append({"role": "user", "content": content})
            elif role == "assistant":
                chat_history.append({"role": "assistant", "content": content})

        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": formatted_prompt}


def prepare_dataset(
    model_path: str = "google/gemma-2b-it",
    hf_user: str = "hf_user",
    num_tokens: int = 800_000_000,
    context_size: int = 1024,
    hf_token: str | None = None,
    private: bool = False
):


    """Generate tokenized dataset, push to huggingface."""
    dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")

    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              token=hf_token)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    dataset = dataset.map(Reasoner(tokenizer).prompt).shuffle(seed=42)

    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=tokenizer,
        streaming=False,
        max_length=context_size,
        column_name="text",
        add_bos_token=False
    )

    num_samples = min(math.ceil(num_tokens / context_size), len(token_dataset))
    token_dataset = token_dataset.select(random.sample(range(len(token_dataset)), num_samples))
    print(">>> Tokens in the dataset = {}".format(len(token_dataset) * context_size))

    repo_id = os.path.join(hf_user, os.path.basename(model_path) + "-lmsys-subset-tokenized-v2")
    token_dataset_dict = DatasetDict({"train": token_dataset})
    token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
