import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Transformer Autoencoder
# -----------------------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        """
        Params:
        - input_dim: number of features per time step. If time series has 8 sensor readings at each time step, then this is 8.
        - model_dim: internal embedding size; dimension of the hidden representations inside the Transformer. 
            *MUST BE DIVISIBLE BY num_heads*
            Typical values: 32-128 (small tasks); 256-512 (medium tasks); 768+ (large scale)
            Higher -> more capacity to learn complex dependencies, but more memory and compute.
            Lower -> faster training and less overfitting risk, but less expressive power. 
        - num_heads: number of attention heads in the multi-head self-attention mechanism. 
            Each head learns to attend to different aspects of the sequence (one might focus on long-term trends, the other on short-term dependencies).
            Typical values: 2, 4, or 8 depending on model_dim.
        - num_layers: number of stacked Transformer blocks (encoder layers and decoder layers). 
            Each layer includes multi-head attention, FF sub-network, residual connections + LayerNorm.
            Typical values: 1-2 (shallow, can capture local dependencies); 4-12+ (deep, can capture complex global structures)
            More layers -> greater modeling depth and abstraction, better representational power, can overfit small datasets.
            Fewer layers -> faster and easier to train but may underfit complex data.
        """
        super(TransformerAutoencoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # Prepare input
        x = self.input_proj(x)  # (batch, seq_len, model_dim)
        x = self.pos_encoding(x)

        # Transformer expects shape (seq_len, batch, dim)
        x = x.permute(1, 0, 2)

        # Encode
        memory = self.encoder(x)

        # Decode
        out = self.decoder(x, memory)

        # Back to original shape
        out = out.permute(1, 0, 2)
        out = self.output_proj(out)
        return out


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on
        # position and i (dimension)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    batch_size = 32
    seq_len = 50
    input_dim = 8

    model_dim = 64
    num_heads = 4
    num_layers = 2

    model = TransformerAutoencoder(input_dim, model_dim, num_heads, num_layers)
    x = torch.randn(batch_size, seq_len, input_dim)

    reconstructed = model(x)

    print("Input shape:", x.shape)
    print("Reconstructed shape:", reconstructed.shape)

    # Example training step
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss = criterion(reconstructed, x)
    loss.backward()
    optimizer.step()

    print("Training step completed. Loss:", loss.item())
