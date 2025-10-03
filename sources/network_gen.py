import data
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim

####################
# 2) Build S using PCC on non-zero overlaps (vectorized-ish)
####################
def gene_pcc_nonzero(X, min_overlap_frac=0.05):
    n, m = X.shape
    S = np.zeros((n, n), dtype=np.float32)
    nonzero = (X > 0).astype(int)
    for i in range(n):
        xi = X[i]
        ei = nonzero[i]
        for j in range(i+1, n):
            xj = X[j]
            ej = nonzero[j]
            # overlap where both non-zero
            mask = (ei & ej).astype(bool)
            if mask.sum() < max(1, int(min_overlap_frac * m)):
                val = 0.0
            else:
                # PCC of xi[mask], xj[mask]
                a = xi[mask]; b = xj[mask]
                if a.std() == 0 or b.std() == 0:
                    val = 0.0
                else:
                    val = np.corrcoef(a, b)[0,1]
                    if np.isnan(val): val = 0.0
            S[i,j] = S[j,i] = val
    return S

####################
# 3) Threshold -> adjacency
####################
def threshold_adj(S, theta=0.35):
    A = S.copy()
    A[np.abs(A) < theta] = 0.0
    # optionally keep sign or take abs; here keep weighted signed edges
    return A

####################
# 4) Normalize adjacency for GCN
####################
def normalize_adj(A):
    # add self-loop
    A_hat = A + np.eye(A.shape[0], dtype=A.dtype)
    deg = np.array(A_hat.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

####################
# 5) Node features: reduce gene x cell -> d-dim via PCA
####################
def gene_features_pca(X, d=128):
    # X: n_genes x n_cells
    pca = PCA(n_components=d)
    H = pca.fit_transform(X)  # gene-level features
    return H

####################
# 6) Simple GCN encoder-decoder (2-layer encoder, linear decoder)
####################
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, H, A_hat):
        # H: n x d, A_hat: n x n (torch)
        return torch.relu(self.lin(torch.matmul(A_hat, H)))

class GCN_AE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.enc1 = GCNLayer(in_dim, hid_dim)
        self.enc2 = GCNLayer(hid_dim, out_dim)
        # decoder: map embedding back to cell expression via linear layer
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)  # we'll decode per cell separately
        )
    def forward(self, H, A_hat, decode_cells):
        Z = self.enc1(H, A_hat)
        Z = self.enc2(Z, A_hat)  # n x out_dim
        # decode: for each cell, project Z -> scalar predicted expression for that cell
        # decode_cells: a matrix of shape (n, m_cells) of gene-level targets (we'll process per cell)
        # For efficiency do linear on Z to get per-gene scalar then compare to decode_cells
        out = self.decoder(Z).squeeze(-1)  # n
        return out, Z

####################
# 7) Training loop (simplified): we train to reconstruct per-gene aggregate (or per-cell via loop)
####################
def train_model(X, A, d_feat=128, hid=64, emb=32, lr=1e-3, epochs=200, device='cpu'):
    # X: n x m
    n, m = X.shape
    H0 = gene_features_pca(X, d=d_feat)  # n x d
    A_hat = normalize_adj(A)  # n x n
    # to torch
    H_t = torch.tensor(H0, dtype=torch.float32, device=device)
    A_t = torch.tensor(A_hat, dtype=torch.float32, device=device)
    model = GCN_AE(in_dim=d_feat, hid_dim=hid, out_dim=emb).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    # We will reconstruct per-cell by training on random mini-batches of cells:
    for ep in range(epochs):
        model.train()
        loss_epoch = 0.0
        # iterate cells in mini-batches
        for start in range(0, m, 64):
            end = min(m, start+64)
            Xbatch = torch.tensor(X[:, start:end], dtype=torch.float32, device=device)  # n x bs
            # create mask M_train: use observed values as supervision (or randomly mask some observed to simulate dropout)
            M = (Xbatch > 0).float()
            # forward
            pred_scalar, Z = model(H_t, A_t, Xbatch)
            # pred_scalar is n -> single scalar; here we need per-cell predictions.
            # Simplest way: treat decoder as predicting average expression across cells.
            # A better variant: make decoder produce per-cell by conditioning on cell embedding (omitted for brevity).
            # For demo, we compute loss against gene-wise mean in batch:
            target = Xbatch.mean(dim=1)  # n
            loss = ((pred_scalar - target)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_epoch += loss.item()
        if ep % 20 == 0:
            print(f"Epoch {ep} loss {loss_epoch:.4f}")
    return model, A_hat

d = data.load_data("./data/sim2.counts.normalized.csv")
S = gene_pcc_nonzero(d.values, min_overlap_frac=0.05)
A = threshold_adj(S, theta=0.35)
print('Training GCN model...')
model, A_hat = train_model(d.values, A, d_feat=128, hid=64, emb=32, lr=1e-3, epochs=200, device='cpu')  
print('Training complete.')
data.save_data(pd.DataFrame(A_hat), "./data/sim2_gene_graph.csv")
torch.save(model.state_dict(), "./models/GCN/gcn_ae_model.pt")