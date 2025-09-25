import fire
import json
import os

from datasets import DatasetDict, load_dataset, concatenate_datasets


class Reasoner:
    """Simple class derived from ambik to prepare prompts, modified to support gemma."""
    gemma_chat_template = """{system}<｜User｜>{user}<｜Assistant｜><think>
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


def prepare_openthoughts_subset(
    num_samples: int = 10_000,
    hf_user: str = "hf_user",
    hf_token: str | None = None,
    private: bool = False,
    domains: tuple = ('code', 'math', 'puzzle', 'biology', 'chemistry', 'physics')
):
    """Generate a subset of openthoughts-114k dataset stratified (if possible) 
      by domains and in the deepseek format.
    """
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", 
                           "metadata", 
                           split="train", 
                           token=hf_token)
    dataset = dataset.filter(lambda sample: sample['domain'] in domains)
    dataset = dataset.add_column("orig_idx", list(range(len(dataset))))

    samples_per_domain = num_samples // len(domains)

    subsets = []
    selected_indices = set()
    for domain in domains:
        domain_ds = dataset.filter(lambda x: x["domain"] == domain).shuffle(seed=42)
        
        # Get as many samples as available up to `samples_per_domain`
        sample_from_domain = min(len(domain_ds), samples_per_domain)
        domain_subset = domain_ds.select(range(sample_from_domain))
        subsets.append(domain_subset)
        print(f">>> samples in {domain} = {sample_from_domain}")

        selected_indices.update(domain_subset["orig_idx"])

    current_total = sum(map(len, subsets))
    if current_total < num_samples:
        additional_needed = num_samples - current_total
        print("Extending, additional samples needed:", additional_needed)
        
        remaining_dataset = dataset.filter(lambda x: x["orig_idx"] not in selected_indices)
        additional_samples = remaining_dataset.select(range(additional_needed))
        subsets.append(additional_samples)
    
    final_dataset = concatenate_datasets(subsets)
    final_dataset = final_dataset.remove_columns("orig_idx")
    final_dataset = final_dataset.map(Reasoner().prompt)

    repo_id = os.path.join(hf_user, f"OpenThoughts-{num_samples//1000}k-DeepSeek-R1")
    final_dataset = DatasetDict({"train": final_dataset})
    final_dataset.push_to_hub(repo_id, token=hf_token, private=private)

if __name__ == "__main__":
    fire.Fire(prepare_openthoughts_subset)
