# models_autoenc.py
import math
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# -------------------------
# LSTM Autoencoder
# -------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, latent_size=64, num_layers=2, bidirectional=False, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        self.enc_fc = nn.Linear(hidden_size * self.num_directions, latent_size)

        # Decoder
        self.dec_fc = nn.Linear(latent_size, hidden_size * self.num_directions)
        self.decoder_lstm = nn.LSTM(
            hidden_size * self.num_directions, input_size, num_layers=num_layers,
            batch_first=True, bidirectional=False, dropout=dropout
        )
        # final projection (decoder LSTM returns features, but we want 1-dim output)
        # we made decoder LSTM output dimens = input_size already, keep as is

    def forward(self, x):
        # x: (batch, seq_len, 1)
        batch, seq_len, _ = x.shape

        # Encoder
        enc_out, (hn, cn) = self.encoder_lstm(x)  # enc_out: (batch, seq_len, hidden*dir)
        # take last time-step hidden (for LSTM with batch_first): use hn from last layer
        if self.bidirectional:
            # hn shape: (num_layers*2, batch, hidden)
            hn_cat = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden*2)
        else:
            hn_cat = hn[-1]  # (batch, hidden)
        latent = self.enc_fc(hn_cat)  # (batch, latent_size)

        # Decoder: expand latent to initial hidden for decoder LSTM (we'll feed zero inputs)
        dec_h0 = self.dec_fc(latent)  # (batch, hidden*num_directions)
        # split into layers: set all layers' hidden to dec_h0 repeated for num_layers
        # build h0 and c0 of shape (num_layers, batch, hidden_size*1)
        dec_hidden = dec_h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_size*num_directions)
        dec_cell = torch.zeros_like(dec_hidden)

        # Prepare decoder inputs: zeros (teacher forcing would be possible if available)
        dec_inputs = torch.zeros(batch, seq_len, self.input_size, device=x.device, dtype=x.dtype)

        dec_out, _ = self.decoder_lstm(dec_inputs, (dec_hidden, dec_cell))
        # dec_out: (batch, seq_len, input_size)
        # If decoder returns exactly input_size dims, done. If not, add projection:
        if dec_out.shape[-1] != self.input_size:
            dec_out = nn.Linear(dec_out.shape[-1], self.input_size).to(x.device)(dec_out)
        return dec_out, latent


# -------------------------
# Transformer Autoencoder
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd dimension guard
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, latent_size=64, dropout=0.1, max_seq_len=2000):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.latent_size = latent_size
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)

        # Encoder stack
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Bottleneck
        self.bottleneck = nn.Linear(d_model, latent_size)
        self.bottleneck_inv = nn.Linear(latent_size, d_model)

        # Decoder stack (we will expand latent back to sequence length via a linear and refine with transformer layers)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection to 1-dim
        self.output_proj = nn.Linear(d_model, input_size)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        batch, seq_len, _ = x.shape
        assert seq_len <= self.max_seq_len, f"seq_len({seq_len}) > max_seq_len({self.max_seq_len})"

        # encode
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)
        enc = self.encoder(h)  # (batch, seq_len, d_model)

        # pool to latent (mean pooling)
        pooled = enc.mean(dim=1)  # (batch, d_model)
        latent = self.bottleneck(pooled)  # (batch, latent_size)

        # decode: expand latent to sequence-length embeddings
        dec_seed = self.bottleneck_inv(latent)  # (batch, d_model)
        # expand to (batch, seq_len, d_model)
        dec_seq = dec_seed.unsqueeze(1).repeat(1, seq_len, 1)
        dec_seq = self.pos_enc(dec_seq)
        dec_refined = self.decoder(dec_seq)  # (batch, seq_len, d_model)
        out = self.output_proj(dec_refined)  # (batch, seq_len, 1)
        return out, latent


# -------------------------
# Training / Eval utilities
# -------------------------
def train_epoch(model, dataloader, optimizer, device, mse_weight=1.0, clip_grad=None):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss(reduction='mean')
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        # batch shape: (batch, seq_len, 1) or list -> handle tensors only
        x = batch.to(device)
        optimizer.zero_grad()
        recon, latent = model(x)
        loss = criterion(recon, x) * mse_weight
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(train_loss=running_loss / ((pbar.n + 1) * dataloader.batch_size))
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss(reduction='none')  # we'll compute per-sample MSE
    all_losses = []
    pbar = tqdm(dataloader, desc="Val", leave=False)
    for batch in pbar:
        x = batch.to(device)
        recon, latent = model(x)
        per_elem = criterion(recon, x)  # (batch, seq_len, 1)
        per_sample_mse = per_elem.mean(dim=(1, 2)).cpu()  # (batch,)
        all_losses.append(per_sample_mse)
    all_losses = torch.cat(all_losses, dim=0)
    return all_losses  # Tensor of per-window MSE


# -------------------------
# Example training script
# -------------------------
if __name__ == "__main__":
    # Example usage:
    # Assumes your project has get_health_divided_loaders imported into scope
    # from src.data.your_module import get_health_divided_loaders
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Replace this with the exact import path if needed:
    try:
        from src.data.train_test_split import get_health_divided_train_test  # to ensure import path exists
        from src.data import BatteryData  # ensure availability
        # The user's helper function in the prompt:
        from src.data.your_loader_module import get_health_divided_loaders  # <-- replace with correct module if needed
    except Exception:
        # If above import doesn't match your project layout, you can directly call the function where it's defined.
        pass

    # hyperparams
    window_size = 1000
    batch_size = 64
    lr = 1e-3
    epochs = 20
    model_type = "transformer"  # change to "lstm" to use the LSTM AE

    # If you have get_health_divided_loaders in scope just call it:
    try:
        train_loader, val_loader, train_ds, val_ds = get_health_divided_loaders(
            window_size=window_size,
            window_skip=1,
            batch_size=batch_size,
            num_workers=4,
            preload=False,
        )
    except NameError:
        raise RuntimeError("Please import or define get_health_divided_loaders in the same namespace before running this script.")

    # instantiate model
    if model_type.lower().startswith("lstm"):
        model = LSTMAutoencoder(input_size=1, hidden_size=256, latent_size=64, num_layers=2, bidirectional=False).to(device)
    else:
        model = TransformerAutoencoder(input_size=1, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, latent_size=64, max_seq_len=window_size).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    best_val_median = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, clip_grad=1.0)
        print(f" Train loss: {train_loss:.6f}")
        val_losses = validate(model, val_loader, device)
        median_val = float(torch.median(val_losses).item())
        mean_val = float(val_losses.mean().item())
        print(f" Val mean MSE: {mean_val:.6e}, median MSE: {median_val:.6e}")

        # simple checkpoint
        ckpt_path = f"ae_{model_type}_epoch{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)

        if median_val < best_val_median:
            best_val_median = median_val
            torch.save({"model_state": model.state_dict()}, f"ae_{model_type}_best.pt")

    # After training, compute per-window reconstruction scores for test set:
    test_losses = validate(model, val_loader, device)
    torch.save({"test_mse": test_losses}, "test_mse_scores.pt")
    print("Saved test MSE scores (per-window) to test_mse_scores.pt")
