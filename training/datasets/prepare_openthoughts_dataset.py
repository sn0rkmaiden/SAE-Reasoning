import fire
import json
import os
import random
import math

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from transformer_lens.utils import tokenize_and_concatenate


class Reasoner:
    """Simple class derived from open-thoughts to prepare prompts, modified to support DeepSeek."""
    deepseek_r1_chat_template = """{system}<｜User｜>{user}<｜Assistant｜><think>
{assistant_reasoning}
</think>
{assistant_solution}"""

    deepseek_r1_system_prompt = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""

    def prompt_code(self, input):
        def format_prompt(x):
            formatted_prompt = ""
            
            data = json.loads(x["test_cases"])
            if not data.get("fn_name"):
                formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
            else:
                formatted_prompt += (
                    "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # noqa
                )

            formatted_prompt += x["problem"]
            if x["starter_code"] is not None:
                data = x["starter_code"]
                data = "\n" + data
                formatted_prompt += data
            return formatted_prompt

        formatted_prompt = format_prompt(input)
        return {"text": self.deepseek_r1_chat_template.format(system=self.deepseek_r1_system_prompt,
                                                              user=formatted_prompt,
                                                              assistant_reasoning=input["deepseek_reasoning"],
                                                              assistant_solution=input["deepseek_solution"])}

    def prompt_math(self, input):
        return {"text": self.deepseek_r1_chat_template.format(system=self.deepseek_r1_system_prompt,
                                                              user=f"Return your final response within \\boxed{{}}. {input['problem']}",
                                                              assistant_reasoning=input["deepseek_reasoning"],
                                                              assistant_solution=input["deepseek_solution"])}

    def prompt_puzzle(self, input):
        return {"text": self.deepseek_r1_chat_template.format(system=self.deepseek_r1_system_prompt,
                                                              user=input['problem'],
                                                              assistant_reasoning=input["deepseek_reasoning"],
                                                              assistant_solution=input["deepseek_solution"])}

    def prompt_science(self, input):
        return {"text": self.deepseek_r1_chat_template.format(system=self.deepseek_r1_system_prompt,
                                                              user=input['problem'],
                                                              assistant_reasoning=input["deepseek_reasoning"],
                                                              assistant_solution=input["deepseek_solution"])}

    def prompt(self, input):
        domain = input['domain']
        
        if domain == 'code':
            return self.prompt_code(input)
        elif domain == 'math':
            return self.prompt_math(input)
        elif domain == 'puzzle':
            return self.prompt_puzzle(input)
        elif domain in ['biology', 'chemistry', 'physics']:
            return self.prompt_science(input)
        else:
            raise NotImplementedError("Do not support format for {}".format(domain))


def prepare_openthoughts_dataset(
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    hf_user: str = "hf_user",
    num_tokens: int = 800_000_000,
    context_size: int = 1024,
    hf_token: str | None = None,
    private: bool = False,
    domains: tuple = ('code', 'math', 'puzzle', 'biology', 'chemistry', 'physics')
):
    """Generate tokenized openthoughts-114k dataset, push to huggingface."""
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", 
                           "metadata", 
                           split="train", 
                           token=hf_token)
    dataset = dataset.filter(lambda sample: sample['domain'] in domains)
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

    repo_id = os.path.join(hf_user, os.path.basename(model_path) + "-OpenThoughts-114k-tokenized")
    token_dataset_dict = DatasetDict({"train": token_dataset})
    token_dataset_dict.push_to_hub(repo_id, token=hf_token, private=private)


if __name__ == "__main__":
    fire.Fire(prepare_openthoughts_dataset)
