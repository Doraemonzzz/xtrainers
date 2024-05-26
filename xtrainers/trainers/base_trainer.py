

class Trainer:
    def __init__(
        self,
        config_dict,
        model,
        tokenizer,
        train_data,
        valid_data,
    ):
        super().__init__()