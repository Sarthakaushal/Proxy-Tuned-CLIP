from config import CFG, TrainingCFG
from dataloader import CLIPDataset, get_transforms
import asyncio
import torch
import pandas as pd
from tqdm import tqdm
# from power_usage import initialize_nvml, get_power_usage, shutdown_nvml
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def make_train_valid_dfs():
    train_dataframe = pd.read_csv(CFG.captions_path_train)
    valid_dataframe = pd.read_csv(CFG.captions_path_val)
    test_dataframe = pd.read_csv(CFG.captions_path_test)
    return train_dataframe, valid_dataframe, test_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms()
    dataset = CLIPDataset(
        dataframe["Filename"].values,
        dataframe["ClassName"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TrainingCFG.batch_size,
        num_workers=TrainingCFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step , pow_usage):
    # initialize_nvml()
    loss_meter = AvgMeter()
    # power_task = get_power_usage()
    # print(f"Power Usage before the start {power_task}")
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(TrainingCFG.device) for k, v in batch.items() 
                    if k != "caption"}
        loss = model(batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        # power_task = get_power_usage()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        # power = power_task
        # print(f"Power Usage: {power} watts")
        # pow_usage.append(power)
    # shutdown_nvml()
    return loss_meter, pow_usage


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(TrainingCFG.device) for k, v in batch.items() 
                    if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

