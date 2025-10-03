import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import pandas as pd

# ---------------------------
# 1️⃣ Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()   # giá trị biểu hiện không âm
        )
    def forward(self, z):
        return self.model(z)

# ---------------------------
# 2️⃣ Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ---------------------------
# 3️⃣ Khởi tạo mô hình
# ---------------------------
X = data.load_data("./data/sim2.counts.normalized.csv").values  # n_genes x n_cells
M = (X > 0).astype(float)  # Mask ma trận (1 nếu biểu hiện > 0, ngược lại 0)
n_genes, n_cells = X.shape
latent_dim = 64
hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Using device:", device)

G = Generator(z_dim=latent_dim, hidden_dim=hidden_dim, output_dim=n_genes).to(device)
D = Discriminator(input_dim=n_genes, hidden_dim=hidden_dim).to(device)

d_opt = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.9))
g_opt = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))

bce = nn.BCELoss()

# ---------------------------
# 4️⃣ Training loop GAN
# ---------------------------
epochs = 1000
for epoch in range(epochs):
    for cell_idx in range(n_cells):
        real_data = torch.tensor(X[:, cell_idx], dtype=torch.float32, device=device)
        mask = torch.tensor(M[:, cell_idx], dtype=torch.float32, device=device)

        # (1) Train Discriminator
        z = torch.randn(latent_dim, device=device)
        fake_data = G(z)

        d_real = D(real_data)
        d_fake = D(fake_data.detach())

        d_loss = bce(d_real, torch.ones_like(d_real)) + \
                 bce(d_fake, torch.zeros_like(d_fake))

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # (2) Train Generator
        d_fake = D(fake_data)
        g_loss = bce(d_fake, torch.ones_like(d_fake))

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}]  D loss: {d_loss.item():.4f}  G loss: {g_loss.item():.4f}")

# ---------------------------
# 5️⃣ Bù khuyết dữ liệu
# ---------------------------
imputed_data = X.copy()
for cell_idx in range(n_cells):
    z = torch.randn(latent_dim, device=device)
    fake = G(z).detach().cpu().numpy()
    imputed_data[:, cell_idx] = X[:, cell_idx] * M[:, cell_idx] + fake * (1 - M[:, cell_idx])
data.save_data(pd.DataFrame(imputed_data), "./data/sim2.counts.imputed_gan.csv")
