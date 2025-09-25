import fire
import os
import random
import math

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from transformer_lens.utils import tokenize_and_concatenate


class Reasoner:
    """Simple class to prepare prompts from LMSys-Chat-1M for DeepSeek."""
    def prompt(self, input):
        conversation = input["conversation"]

        prompt = []
        for i, message in enumerate(conversation, start=1):
            if message['role'] == 'system':
                prompt.append(message['content'])
            elif message['role'] == 'user':
                prompt.append('<｜User｜>' + message['content'])
            elif message['role'] == 'assistant':
                prompt.append('<｜Assistant｜>' + message['content'])
                if i != len(conversation):
                    prompt.append('<｜end▁of▁sentence｜>')

        return {"text": ''.join(prompt)}


def prepare_lmsys_dataset(
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    hf_user: str = "hf_user",
    num_tokens: int = 800_000_000,
    context_size: int = 1024,
    hf_token: str | None = None,
    private: bool = False
):
    """Generate tokenized lmsys-chat-1m dataset, push to huggingface."""
    dataset = load_dataset("lmsys/lmsys-chat-1m",
                           split="train",
                           token=hf_token)
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

    repo_id = os.path.join(hf_user, os.path.basename(model_path) + "-lmsys-chat-1m-tokenized")
    token_dataset_dict = DatasetDict({"train": token_dataset})
    token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)


if __name__ == "__main__":
    fire.Fire(prepare_lmsys_dataset)
