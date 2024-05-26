import argparse
import os

import datasets
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from yaml.loader import SafeLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the YAML file"
    )
    return parser.parse_args()


def get_configs(config_file):
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    return config_dict


CONFIG_TO_CONFIG_CLASS = {"llama": LlamaConfig}

CONFIG_TO_MODEL_CLASS = {
    "causal_lm": AutoModelForCausalLM,
}


def get_model(config_dict):
    model_config_class = CONFIG_TO_CONFIG_CLASS[config_dict["model_name"]]
    model_config = model_config_class.from_dict(config_dict["model_config"])
    model_class = CONFIG_TO_MODEL_CLASS[config_dict["model_type"]]
    model = model_class.from_config(model_config)
    print(model)


def get_data(config_dict):
    data = datasets.load_from_disk(config_dict["data_dir"])
    train_data = data["train"]
    valid_data = data["validation"]
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config_dict["data_dir"], "tokenizer")
    )

    return tokenizer, train_data, valid_data


def main():
    args = get_args()
    config_dict = get_configs(args.config_file)

    print(config_dict)

    # Get model, data, loss
    get_model(config_dict["model"])
    tokenizer, train_data, valid_data = get_data(config_dict["data"])
    # loss = get_loss(config_file.loss)

    # # Setup trainer
    # trainer = get_trainer(config_file, model, dataloader, loss)

    # # Train
    # trainer.train()


# def get_model(

# ):
#     # get data config

#     # get model

if __name__ == "__main__":
    main()
