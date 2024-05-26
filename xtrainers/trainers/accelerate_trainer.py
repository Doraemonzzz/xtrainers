# The code is change from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

import logging
import math
import os

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator, get_scheduler

logger = get_logger(__name__)
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class AccTrainer:
    def __init__(
        self,
        config_dict,
        model,
        tokenizer,
        train_data,
        valid_data,
    ):
        super().__init__()

        self.train_config_dict = config_dict["training_config"]
        self.optimizer_config_dict = config_dict["optimizer"]
        self.lr_scheduler_config_dict = config_dict["lr_scheduler"]
        self.checkpoints_config_dict = config_dict["checkpoints"]
        self.loss_type = config_dict["loss"]["type"]

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_config_dict[
                "gradient_accumulation_steps"
            ]
        )
        self.total_batch_size = len(train_data)
        # Init dataloader
        train_dataloader, valid_dataloader = self.get_dataloader(train_data, valid_data)
        # Init optimizer
        optimizer = self.get_optimizer(model)
        # Init lr scheduler
        lr_scheduler = self.get_lr_scheduler(optimizer, train_dataloader)

        # Prepare everything with our `accelerator`.
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.valid_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
        )
        self.loss_fn = self.get_loss_fn()
        
    def get_loss_fn(self):
        loss_type = self.loss_type
        if loss_type == "naive_ce":
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss()

    def get_dataloader(self, train_data, valid_data):
        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_data,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=self.train_config_dict["train_batch_size"],
        )
        valid_dataloader = DataLoader(
            valid_data,
            collate_fn=default_data_collator,
            batch_size=self.train_config_dict["valid_batch_size"],
        )

        return train_dataloader, valid_dataloader

    def get_optimizer_class(self):
        optimizer_type = self.optimizer_config_dict["type"]
        if optimizer_type == "adamw":
            return torch.optim.AdamW
        else:
            return torch.optim.AdamW

    def get_optimizer(self, model):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = [
            "bias",
            "layer_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and (not any(nd in n for nd in no_decay))
                ],
                "weight_decay": self.train_config_dict["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and (any(nd in n for nd in no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_class = self.get_optimizer_class()
        optimizer = optimizer_class(
            optimizer_grouped_parameters, lr=self.train_config_dict["learning_rate"]
        )

        return optimizer

    def get_lr_scheduler(self, optimizer, train_dataloader):
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader)
            / self.train_config_dict["gradient_accumulation_steps"]
        )
        max_train_steps = self.train_config_dict.get("max_train_steps", 0)
        if max_train_steps == 0:
            max_train_steps = (
                self.train_config_dict["num_train_epochs"] * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        # Afterwards we recalculate our number of training epochs
        self.max_train_steps = max_train_steps
        self.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_config_dict["type"],
            optimizer=optimizer,
            num_warmup_steps=self.train_config_dict["num_warmup_steps"]
            * self.accelerator.num_processes,
            num_training_steps=max_train_steps
            if overrode_max_train_steps
            else max_train_steps * self.accelerator.num_processes,
        )

        return lr_scheduler
    
    def compute_loss(self, batch, logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :]
        shift_labels = batch["input_ids"][..., 1:]
        # Flatten the tokens
        shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.reshape(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.loss_fn(shift_logits, shift_labels)
        
        return loss

    def train(self):
        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = self.train_config_dict["checkpointing_steps"]

        # Train!
        total_batch_size = (
            self.train_config_dict["train_batch_size"]
            * self.accelerator.num_processes
            * self.train_config_dict["gradient_accumulation_steps"]
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {self.total_batch_size}")
        logger.info(f"  Num Epochs = {self.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.train_config_dict['train_batch_size']}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.train_config_dict['gradient_accumulation_steps']}"
        )
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        completed_steps = 0
        starting_epoch = 0

        # # Potentially load in the weights and states from a previous save
        # if args.resume_from_checkpoint:
        #     if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
        #         checkpoint_path = args.resume_from_checkpoint
        #         path = os.path.basename(args.resume_from_checkpoint)
        #     else:
        #         # Get the most recent checkpoint
        #         dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        #         dirs.sort(key=os.path.getctime)
        #         path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        #         checkpoint_path = path
        #         path = os.path.basename(checkpoint_path)

        #     accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        #     accelerator.load_state(checkpoint_path)
        #     # Extract `epoch_{i}` or `step_{i}`
        #     training_difference = os.path.splitext(path)[0]

        #     if "epoch" in training_difference:
        #         starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        #         resume_step = None
        #         completed_steps = starting_epoch * num_update_steps_per_epoch
        #     else:
        #         # need to multiply `gradient_accumulation_steps` to reflect real steps
        #         resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        #         starting_epoch = resume_step // len(train_dataloader)
        #         completed_steps = resume_step // args.gradient_accumulation_steps
        #         resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, self.train_config_dict["num_train_epochs"]):
            self.model.train()
            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            if False:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader, resume_step
                )
            else:
                active_dataloader = self.train_dataloader
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = self.compute_loss(batch, outputs.logits)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if step % self.train_config_dict["log_interval"] == 0:
                    logger.info(f"step: {step}, loss: {loss}")

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if self.train_config_dict["output_dir"] is not None:
                            output_dir = os.path.join(
                                self.train_config_dict["output_dir"], output_dir
                            )
                        self.accelerator.save_state(output_dir)
                if completed_steps >= self.max_train_steps:
                    break

            self.model.eval()
            losses = []
            for step, batch in enumerate(self.valid_dataloader):
                with torch.no_grad():
                    outputs = self.model(**batch)

                loss = self.compute_loss(batch, outputs.logits)
                losses.append(
                    self.accelerator.gather_for_metrics(
                        loss.repeat(self.train_config_dict["valid_batch_size"])
                    )
                )

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(
                f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}"
            )

            # save every epoch
            output_dir = f"epoch_{epoch}"
            if self.train_config_dict["output_dir"] is not None:
                output_dir = os.path.join(
                    self.train_config_dict["output_dir"], output_dir
                )
            self.accelerator.save_state(output_dir)
