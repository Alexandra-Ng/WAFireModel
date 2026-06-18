"""
Graph convolutional network

First layer is equation for H' = ReLU(A_hat @ H @ W)
Stack 2 layers: each countrys output is dependent on its 2 hop neighborhood
"""

import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    """
    One graph convolution step
    A_hat @ (x W)
    activation applied outside
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim) # learnable W + bias

    def forward(self, x, A_hat):
        return A_hat @ self.lin(x) # Linear map, mix with neighbors

class GCN(nn.Module):
    """
    Two later GCN for per-node regression
    One number per county
    """
    def __init__(self, in_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_hat):
        h = torch.relu(self.gc1(x, A_hat)) # Layer 1 and nonlinearity
        h = self.dropout(h) # Regularise because 39 nodes is small
        out = self.gc2(h, A_hat) # Layer 2 (N, 1)
        return out.squeeze(-1) # (N,)

if __name__ == "__main__":
    from src.main.config import load_config
    from src.data.adjacency import load_adjacency, normalize_adjacency
    from src.data.dataset import load_dataset, FEATURES

    cfg = load_config("configs/default.yaml")
    counties, A = load_adjacency(cfg["graph"]["adjacency_file"])
    A_hat = normalize_adjacency(A)
    _, data, _ = load_dataset(cfg)

    model = GCN(in_dim=len(FEATURES), hidden_dim=cfg["model"]["hidden_dim"])
    X, y, mask = data[2019]
    out = model(X, A_hat) # untrained values are random
    print(model)
    print("X: ", tuple(X.shape), "->out:", tuple(out.shape))
    print("one prediction per county: ", out.shape[0] == len(counties))








