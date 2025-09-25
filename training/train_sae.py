import os
# os.environ["WANDB_API_KEY"] = "YOUR WANDB API KEY"

import fire
import yaml

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner


def load_config(yaml_path: str):
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def main(config_path: str):
    config_dict = load_config(config_path)

    config = LanguageModelSAERunnerConfig(**config_dict)

    print(">>> Start training SAE")
    SAETrainingRunner(config).run()
    print(">>> Finished training SAE")


if __name__ == "__main__":
    fire.Fire(main)
