"""County adjacency graph for GCN as a normalized matrix"""

import json
from pathlib import Path
import numpy as np
import torch

def load_adjacency(path):
    """Read the county to neighbors JSON. Return (counties, A).
    A is the symmetric 0/1 matrix: A[i,j] = 1 if county i & j share a border"""

    adj = json.loads(Path(path).read_text())
    counties = sorted(adj) # fixed node ordering, row/col index
    idx = {c: i for i, c in enumerate(counties)}
    n = len(counties)
    A = np.zeros((n,n), dtype="float32")
    for county, neighbors in adj.items():
        for nb in neighbors:
            A[idx[county], idx[nb]] = 1.0
    A = np.maximum(A, A.T) # To ensure symmetry i,j and j,i
    return counties, A

def normalize_adjacency(A):
    """Return = D^-1/2 (A+I) D^-1/2 GCN propagation matrix
    
    A+I: county keeps its own value (self loop)
    D^-1/2: down weights high degree counties so neighbor averaging doesnt
    blow up when county has a lot of neighbors
    """
    n = A.shape[0]
    A = A + np.eye(n, dtype=np.float32)
    deg = A.sum(axis=1) # degree = neigbor count (+1 self)
    d_inv_sqrt = 1.0/np.sqrt(deg)
    A_hat = d_inv_sqrt[:, None] *A *d_inv_sqrt[None,:]
    return torch.tensor(A_hat)

if __name__ == "__main__":
    counties, A = load_adjacency("src/data/wa_adjacency.json")
    A_hat = normalize_adjacency(A)
    print(f"{len(counties)} counties, {int(A.sum() // 2)} edges")
    print("A_hat shape:", tuple(A_hat.shape))
    print("symmetric:", bool(torch.allclose(A_hat, A_hat.T))) 



