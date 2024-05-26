# The code is change from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

from transformers import default_data_collator, get_scheduler
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
import torch
import math
import os
import logging
from tqdm.auto import tqdm

logger = get_logger(__name__)

class AccTrainer:
    def __init__(
        self,
        config_dict,
        model,
        tokenizer,
        train_data,
        valid_data,
        loss_fn,
    ):
        super().__init__()
        
        train_config_dict = config_dict["training_config"]
        optimizer_config_dict = config_dict["optimizer"]
        lr_scheduler_config_dict = config_dict["lr_scheduler"]
        checkpoints_config_dict = config_dict["checkpoints"]
        
        accelerator = Accelerator(gradient_accumulation_steps=train_config_dict["gradient_accumulation_steps"])
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
            
        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_data, shuffle=False, collate_fn=default_data_collator, batch_size=train_config_dict["train_batch_size"]
        )
        valid_dataloader = DataLoader(
            valid_data, collate_fn=default_data_collator, batch_size=train_config_dict["valid_batch_size"]
        )
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
        # for n, p in model.named_parameters():
        #     print(n, any(nd in n for nd in no_decay))
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": train_config_dict["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=train_config_dict["learning_rate"])
        
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_config_dict["gradient_accumulation_steps"])
        max_train_steps = train_config_dict.get("max_train_steps", 0)
        if max_train_steps == 0:
            max_train_steps = train_config_dict["num_train_epochs"] * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=lr_scheduler_config_dict["type"],
            optimizer=optimizer,
            num_warmup_steps=train_config_dict["num_warmup_steps"] * accelerator.num_processes,
            num_training_steps=max_train_steps if overrode_max_train_steps
            else max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
        )

        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = train_config_dict["checkpointing_steps"]

        # Train!
        total_batch_size = train_config_dict["train_batch_size"] * accelerator.num_processes * train_config_dict["gradient_accumulation_steps"]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_data)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_config_dict['train_batch_size']}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {train_config_dict['gradient_accumulation_steps']}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
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

        for epoch in range(starting_epoch, train_config_dict["num_train_epochs"]):
            model.train()
            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            if False:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                if step % train_config_dict["log_interval"] == 0:
                    logger.info(f"step: {step}, loss: {loss}")

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if train_config_dict["output_dir"] is not None:
                            output_dir = os.path.join(train_config_dict["output_dir"], output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= max_train_steps:
                    break

            model.eval()
            losses = []
            for step, batch in enumerate(valid_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(train_config_dict["valid_batch_size"])))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            # save every epoch
            output_dir = f"epoch_{epoch}"
            if train_config_dict["output_dir"] is not None:
                output_dir = os.path.join(train_config_dict["output_dir"], output_dir)
            accelerator.save_state(output_dir)
        
    def get_optimizer(self, optimizer_type):
        if optimizer_type == "adamw":
            return torch.optim.AdamW
        else:
            return torch.optim.AdamW