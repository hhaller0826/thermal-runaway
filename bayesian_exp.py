import os
import math
import random
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.data import get_divided_loaders
from src.models import TransformerAutoencoder

# -------------------------
# Model training & eval
# -------------------------
def train_epoch(model, dataloader, optimizer, device, epoch_num='', mse_weight=1.0, clip_grad=None):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss(reduction='mean')

    pbar = tqdm(dataloader, desc=f'Train epoch {epoch_num}', leave=True)
    for batch in pbar:
        x = batch.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x) * mse_weight
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        running_loss += loss.item() * x.size(0)

        pbar.set_postfix(train_loss=running_loss / ((pbar.n + 1) * dataloader.batch_size))
        torch.mps.empty_cache()
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device='mps', epoch_num=''):
    model.eval()
    criterion = nn.MSELoss(reduction='mean')
    running_loss = 0.0
    torch.mps.empty_cache()

    pbar = tqdm(dataloader, desc=f"Validate epoch {epoch_num}", leave=True)
    with torch.no_grad():
        for batch in pbar:
            x = batch.to(device)
            recon = model(x)
            recon_cpu = recon.detach().to("cpu")
            del x, recon

            loss = criterion(recon_cpu, batch)
            running_loss += loss.item() * batch.size(0)

            pbar.set_postfix(train_loss=running_loss / ((pbar.n + 1) * dataloader.batch_size))
            
            torch.mps.empty_cache()
            
    return running_loss / len(dataloader.dataset)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

# -------------------------
# Optuna Objective
# -------------------------
def objective(trial):
    set_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    window_skip = trial.suggest_int("window_skip", 300, 800, step=100)
    window_size = trial.suggest_int("window_size", 500, 1500, step=500)
    train_loader, val_loader = get_divided_loaders(
        test_size=0.5,
        window_skip=window_skip, 
        window_size=window_size,
        preload=True
    )

    transformer_model = TransformerAutoencoder(
        input_dim=1,      # number of features per timestep
        model_dim=64,     # hidden embedding dimension
        num_heads=4,      # parallel attention heads
        num_layers=2,     # stacked encoder/decoder layers
    ).to(device)

    optimizer = optim.AdamW(transformer_model.parameters(), lr=1e-3)

    max_epochs = 3
    best_val = float("inf")

    for epoch in range(1,1+max_epochs):
        train_loss = train_epoch(transformer_model, 
                                 dataloader=train_loader, 
                                 optimizer=optimizer, 
                                 device=device, 
                                 epoch_num=epoch,
                                 clip_grad=1.0)
        val_loss = evaluate(transformer_model, dataloader=val_loader, device=device)

        # Report intermediate result
        trial.report(val_loss, epoch)

        # Pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        best_val = min(best_val, val_loss)

    return best_val

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="transformer_mps_opt",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1) # PRUNING AVAILABILITY
    )

    study.optimize(objective, n_trials=3)

    print("Best trial:", study.best_trial.value)
    print("Best params:", study.best_trial.params)