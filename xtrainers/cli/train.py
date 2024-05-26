import argparse
import os

import datasets
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from yaml.loader import SafeLoader

VOCAB_BASE = 8


def convert_to_multiple_of_base(n, base):
    return base * ((n + base - 1) // base)


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


def get_model(config_dict, tokenizer):
    model_config_class = CONFIG_TO_CONFIG_CLASS[config_dict["model_name"]]
    config_dict["model_config"]["vocab_size"] = len(tokenizer)
    model_config = model_config_class.from_dict(config_dict["model_config"])
    model_class = CONFIG_TO_MODEL_CLASS[config_dict["model_type"]]
    model = model_class.from_config(model_config)

    # resize the embedding dim
    embedding_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(
        convert_to_multiple_of_base(embedding_size, VOCAB_BASE)
    )
    print(model)


def get_data(config_dict):
    data = datasets.load_from_disk(config_dict["data_dir"])
    train_data = data["train"]
    valid_data = data["validation"]
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config_dict["data_dir"], "tokenizer")
    )

    return tokenizer, train_data, valid_data


def get_loss(config_dict):
    if config_dict["loss_type"] == "naive_ce":
        return nn.CrossEntropyLoss
    else:
        return nn.CrossEntropyLoss


def main():
    args = get_args()
    config_dict = get_configs(args.config_file)

    print(config_dict)

    # Get tokenizer, data, model, loss
    tokenizer, train_data, valid_data = get_data(config_dict["data"])
    get_model(config_dict["model"], tokenizer)
    get_loss(config_dict["loss"])

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
