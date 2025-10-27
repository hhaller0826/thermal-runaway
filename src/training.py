import torch
from tqdm.auto import tqdm
from src.models.nn_autoencoders import *


def train_epoch(model, dataloader, device, mse_weight=1.0, clip_grad=None):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pbar = tqdm(dataloader, desc="Train", leave=False)
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
    return running_loss / len(dataloader.dataset)
