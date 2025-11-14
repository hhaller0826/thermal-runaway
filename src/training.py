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


def validate(model, dataloader, device='mps'):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_losses = []
    torch.mps.empty_cache()

    pbar = tqdm(dataloader, desc="Validate", leave=False)
    with torch.no_grad():
        for batch in pbar:
            x = batch.to(device)
            recon = model(x)
            recon_cpu = recon.detach().to("cpu")
            del x, recon
            
            per_elem = criterion(recon_cpu, batch)  # (batch, seq_len, 1)
            per_sample_mse = per_elem.mean(dim=(1, 2)).cpu()  # (batch,)
            all_losses.append(per_sample_mse)

            torch.mps.empty_cache()
            
    all_losses = torch.cat(all_losses, dim=0)
    return all_losses
