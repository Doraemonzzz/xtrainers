import argparse
import yaml
from yaml.loader import SafeLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML file")
    return parser.parse_args()

def get_configs(config_file):
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=SafeLoader)
        
    return config_dict

def main():
    args = get_args()
    config_dict = get_configs(args.config_file)

    print(config_dict)

    # Get model, dataloader, loss
    # model = get_model(config_file.model)
    assert False
    # dataloader = get_dataloader(config_file.dataloader)
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