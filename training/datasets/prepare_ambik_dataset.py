import fire
import os
import random
import math

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from transformer_lens.utils import tokenize_and_concatenate


class Reasoner:
    """Simple class to prepare prompts from AmbiK for Gemma."""
    def prompt(self, input):

        env = input['environment_full']
        ambig = input['ambiguous_task']

        prompt = f"Environment: {env}\nInstruction: {ambig}\nWhat should you do next? Ask a question if needed."

        return {"text": ''.join(prompt)}


def prepare_ambik_dataset(
    model_path: str = "google/gemma-2b-it",
    hf_user: str = "hf_user",
    num_tokens: int = 800_000_000,
    context_size: int = 1024,
    hf_token: str | None = None,
    private: bool = False
):
    """Generate tokenized AmbiK dataset, push to huggingface."""
    raw_url = "https://raw.githubusercontent.com/cog-model/AmbiK-dataset/main/ambik_dataset/ambik_calib_100.csv"
    dataset = load_dataset("csv", data_files=raw_url, split="train")
    dataset = dataset.map(Reasoner().prompt).shuffle(seed=42)

    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    tokenizer = AutoTokenizer.from_pretrained(model_path, 
                                              trust_remote_code=True, 
                                              token=hf_token)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

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

    repo_id = os.path.join(hf_user, os.path.basename(model_path) + "-ambik-tokenized")
    token_dataset_dict = DatasetDict({"train": token_dataset})
    token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)


if __name__ == "__main__":
    fire.Fire(prepare_ambik_dataset(hf_user="snork-maiden"))
