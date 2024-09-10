# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
import pandas as pd
import numpy as np
import random
import gc

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.nn.functional as F
from llama_recipes.inference.model_utils import load_model, load_peft_model

from llama_recipes.utils.config_utils import (
    get_dataloader_kwargs,
)
from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
import wandb
from llama_recipes.utils.evaluate import Evaluation

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,val_dataloader, test_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None, evaluate=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])


    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_train_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    lens = batch["lens"]
                    batch = {key: value for key, value in batch.items() if key != "lens" and key != "examples" and key != "examples_mask"}
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        outputs = model(**batch)
                        logits = outputs.logits
                        losses = []
                        for i in range(logits.size(0)): 
                            logit = logits[i, lens[i]-1:-1, :].contiguous()
                            label = batch['labels'][i, lens[i]:].contiguous()
                            loss = F.cross_entropy(logit.view(-1, logit.size(-1)), label.view(-1))
                            losses.append(loss)
                        loss = torch.mean(torch.stack(losses))
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_train_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if train_config.run_validation and total_train_steps % train_config.logging_steps == 0:
                        print("save model!!")
                        model.eval()
                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                save_peft_checkpoint(model, train_config.output_dir)
                                print("save success!")
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        print(f"checkpoint_time:{checkpoint_end_time}")

                    # if total_train_steps % train_config.logging_steps == 0:
                    #     if train_config.enable_fsdp:
                    #         dist.barrier()
                    #     # generate_metrics(model, train_config, val_dataloader, local_rank, tokenizer, wandb_run, total_train_steps)
                    #         print("begin log")
                    #         inference(model,tokenizer, train_config, local_rank, wandb_run, total_train_steps)
                    #         print("end log")
                    #         torch.cuda.empty_cache()
                    #         gc.collect()

                    model.train()
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            checkpoint_start_time = time.perf_counter()
            if train_config.save_model:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    save_peft_checkpoint(model, train_config.output_dir)
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

    return results

def compute_metrics(model, train_config, dataloader, local_rank, tokenizer, wandb_run, evaluate, total_train_steps, type_name):
    print("computing metrics!!")
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_targets = []
    total_eval_steps = 0
    eval_loss = 0.0
    dataloader_list = list(dataloader)
    num_samples = min(100, len(dataloader_list)-1)
    random_subset = random.sample(dataloader_list, num_samples)
    start_time = time.time()
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(random_subset, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            lens = batch["lens"]
            batch = {key: value for key, value in batch.items() if key != "lens" and key != "examples" and key != "examples_mask"}
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            labels = batch["labels"].to(batch["input_ids"].device)
            with torch.no_grad():
            # Forward pass and compute loss
                outputs = model(**batch)
                logits = outputs.logits
                losses = []
                for i in range(logits.size(0)): 
                    logit = logits[i, lens[i]-1:-1, :].contiguous()
                    label = batch['labels'][i, lens[i]:].contiguous()
                    loss = F.cross_entropy(logit.view(-1, logit.size(-1)), label.view(-1))
                    losses.append(loss)
                loss = torch.mean(torch.stack(losses))
                eval_loss += loss.detach().float()
            preds = torch.argmax(outputs.logits, -1)
            predictions = []
            targets = []
            for i in range(preds.size(0)):
                pred = preds[i,lens[i]-1:-1].cpu().numpy()
                label = labels[i,lens[i]:].cpu().numpy()
                predictions.append(pred)
                targets.append(label)
            predictions = np.where((predictions != -100) & (predictions != 12809), predictions, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            targets = np.where(targets != -100, targets, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(targets, skip_special_tokens=True)
            # print(f"decoded_preds:{decoded_preds}")
            # print(f"decoded_labels:{decoded_labels}")
            eval_preds.extend(decoded_preds)
            eval_targets.extend(decoded_labels)
    if train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        eval_loss = eval_loss / world_size
    end_time = time.time()
    runtime = end_time - start_time
    samples_per_second = train_config.num_test / runtime
    steps_per_second = total_eval_steps / runtime
    mean_loss = eval_loss / total_eval_steps
    if wandb_run:
        wandb_run.log({
            f"{type_name}/samples_per_second": samples_per_second,
            f"{type_name}/steps_per_second": steps_per_second,
            f"{type_name}/loss": mean_loss,
            f"{type_name}/runtime": runtime,
        })
    metrics = evaluate.compute(eval_preds, eval_targets)
    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    if wandb_run:
        wandb_run.log({f"{type_name}/{k}": v for k, v in mean_metrics.items()})
    torch.cuda.empty_cache()
    gc.collect()


def inference(model, tokenizer, train_config,local_rank, wandb_run,total_train_steps):
    model_name = "/scratch/qmei_root/qmei/xziyang/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/1d54af340dc8906a2d21146191a9c184c35e47bd"
    peft_model = train_config.output_dir
    # model = load_model(model_name, quantization=None, use_fast_kernels="False")
    model.eval()
    model = load_peft_model(model, peft_model)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    dataloader_list = list(test_dataloader)
    random_subset = random.sample(dataloader_list, min(10, len(dataloader_list)))
    decoded_preds = []
    decoded_labels = []
    torch.cuda.empty_cache()
    gc.collect()
    for batch in random_subset:
        lens = batch["lens"]
        batch = {key: value for key, value in batch.items() if key != "lens" }
        new_batch = {
            "input_ids": batch["examples"],
            "attention_mask": batch["examples_mask"]
        }
        for key in new_batch.keys():
            if train_config.enable_fsdp:
                new_batch[key] = new_batch[key].to(local_rank)
        for key in batch.keys():
            if train_config.enable_fsdp:
                batch[key] = batch[key].to(local_rank)
        
        labels = batch["labels"].to(batch["input_ids"].device)[0]
        # model.to(batch["input_ids"].device)
        with torch.no_grad():
            outputs = model.generate(
                **new_batch,
                max_new_tokens=100,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(outputs[0][lens.item():].cpu(), skip_special_tokens=True)
        labels_np = labels.cpu().numpy()
        labels = np.where(labels_np != -100, labels_np, tokenizer.pad_token_id)
        decoded_label = tokenizer.decode(labels[lens.item():], skip_special_tokens=True)
        print(f"output_text:{output_text}")
        print(f"decoded_label:{decoded_label}")
        decoded_preds.append(output_text)
        decoded_labels.append(decoded_label)
        torch.cuda.empty_cache()
        gc.collect()

    print(f"pred_len:{len(decoded_preds)}")
    print(f"target_len:{len(decoded_labels)}")
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier() 
    predictions_df = pd.DataFrame({'generate_predictions': decoded_preds, 'labels': decoded_labels})
    predictions_df["step"] = total_train_steps
    if wandb_run:
        print("wandb log!")
        records_table = wandb.Table(dataframe=predictions_df)
        wandb_run.log({"predictions by generate": records_table})
    else:
        print("wandb_run is None. Skipping the logging step.")
    del decoded_preds, decoded_labels, new_batch, outputs, output_text, labels
    torch.cuda.empty_cache()
    gc.collect()
    dist.barrier() 


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and torch.version.cuda >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
