from dataloader import CLIPDataset, get_transforms
from config import TrainingCFG, TextEncCFG, ExpertModelImgCFG, TargetModelImgCFG
from config import CFG as cfg
from utils import make_train_valid_dfs, build_loaders, train_epoch, valid_epoch
from model import CLIPModel

from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np
import torch
import itertools
from tqdm import tqdm



def main():
    status =2 # 1: for training expert and anti-expert model; 2 for training target model
    train_df, valid_df, _ = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    if status ==1:
        CFG = ExpertModelImgCFG
    else:
        CFG = TargetModelImgCFG
    print(f'#> Training the CLIP Model with:-\n\t-- Img Enc : {CFG.model} -- Img Enc : {TextEncCFG.model}')
    model = CLIPModel(image_embedding=CFG.image_embedding).to(TrainingCFG.device)
    
    params = [
        {"params": model.image_encoder.parameters(),
            "lr": TrainingCFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), 
            "lr": TrainingCFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), 
            model.text_projection.parameters()), 
            "lr": TrainingCFG.head_lr, 
            "weight_decay": TrainingCFG.weight_decay}
    ]
    if CFG == TargetModelImgCFG:
        torch.save(model.state_dict(), f"{cfg.model_path}base_target_model_{CFG.model}.pt")
        print("Saved Target Model!")
    else:
        torch.save(model.state_dict(), f"{cfg.model_path}anti-expert_{CFG.model}.pt")
        print("Saved Anti-expert Model!")
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=TrainingCFG.patience,
        factor=TrainingCFG.factor)
    step = "epoch"

    best_loss = float('inf')
    
    training_loss_tracker = []
    val_train_loss_dist = [100]
    trainable = True
    epoch = 0
    pow_usage = []
    while(trainable and epoch<=TrainingCFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, pow_usage = train_epoch(model, train_loader, 
                                optimizer, lr_scheduler, step, pow_usage)
        training_loss_tracker.append(train_loss.avg)
        if len(training_loss_tracker)>2:
            if training_loss_tracker[-3] < sum(training_loss_tracker[-3:])/3:
                print('Training ends, training loss increases')
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f"{cfg.model_path}tuned_CLIP_{CFG.model}_{TrainingCFG.epochs}.pt")
            print("Saved Best Model!")
        delta = train_loss.avg - valid_loss.avg
        if delta <val_train_loss_dist[-1]*.04:
            print(f"Delta = {delta} threshold = {val_train_loss_dist[-1]*.04}")
            val_train_loss_dist.append(delta)
        lr_scheduler.step(valid_loss.avg)
        epoch+=1

if __name__ == "__main__":
    main()