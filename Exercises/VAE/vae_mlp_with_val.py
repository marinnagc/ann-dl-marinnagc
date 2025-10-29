import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

# ======= CONFIG =======
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 2
LR = 1e-3
DATASET = 'mnist'  # ou 'fashion'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './vae_mlp_outputs_with_tsne_and_val'
os.makedirs(SAVE_DIR, exist_ok=True)

# ======= DATA =======
transform = transforms.ToTensor()
ds = datasets.MNIST if DATASET == 'mnist' else datasets.FashionMNIST
full_ds = ds(root='./data', train=True, download=True, transform=transform)

val_size = int(0.1 * len(full_ds))
train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ======= MODEL =======
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, LATENT_DIM)
        self.fc_logvar = nn.Linear(400, LATENT_DIM)
        self.fc2 = nn.Linear(LATENT_DIM, 400)
        self.fc3 = nn.Linear(400, 28*28)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

model = VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======= LOSS =======
def vae_loss(recon_x, x, mu, logvar):
    # reconstrução (BCE pixel a pixel)
    BCE = F.binary_cross_entropy(
        recon_x,
        x.view(-1, 28*28),
        reduction='sum'
    )
    # termo KL para regularizar o espaço latente ~ N(0, I)
    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return BCE + KLD

# ======= FUNÇÃO AUXILIAR: PLOT DO ESPAÇO LATENTE =======
def plot_latent_space(model, loader, device, save_dir, latent_dim):
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu, _ = model.encode(x.view(-1, 28*28))
            zs.append(mu.cpu())
            ys.append(y)

    zs = torch.cat(zs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()

    # se o latente já é 2D, usa direto; senão faz t-SNE
    if latent_dim == 2:
        z_plot = zs
    else:
        z_plot = TSNE(n_components=2).fit_transform(zs)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(
        z_plot[:,0],
        z_plot[:,1],
        c=ys,
        cmap='tab10',
        s=5
    )
    plt.colorbar(scatter)
    plt.title("Latent space (mu)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latent_scatter.png")
    plt.close()

# ======= TRAIN + VALIDATION LOOP =======
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # ---------- TREINO ----------
    model.train()
    running_train_loss = 0.0
    for x, _ in train_loader:
        x = x.to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # ---------- VALIDAÇÃO ----------
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            vloss = vae_loss(recon, x, mu, logvar)
            running_val_loss += vloss.item()

    avg_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    print(f"[Epoch {epoch+1}/{EPOCHS}] "
          f"train_loss={avg_train_loss:.4f}  "
          f"val_loss={avg_val_loss:.4f}")

    # ---------- VISUALIZAÇÕES DA ÉPOCA ----------
    with torch.no_grad():
        # comparação input vs reconstrução
        x_val, _ = next(iter(val_loader))
        x_val = x_val.to(DEVICE)
        recon, _, _ = model(x_val)
        recon = recon.view(-1, 1, 28, 28)

        comparison = torch.cat([x_val[:8], recon[:8]])
        utils.save_image(
            comparison.cpu(),
            f"{SAVE_DIR}/recon_epoch_{epoch+1}.png",
            nrow=8
        )

        # amostras puras do espaço latente
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        sample = model.decode(sample).view(-1, 1, 28, 28)
        utils.save_image(
            sample.cpu(),
            f"{SAVE_DIR}/sample_epoch_{epoch+1}.png",
            nrow=8
        )

# ======= CURVA DE LOSS (TREINO vs VAL) =======
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title("VAE Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/loss_curve.png")
plt.close()

# ======= ESPAÇO LATENTE FINAL =======
plot_latent_space(model, val_loader, DEVICE, SAVE_DIR, LATENT_DIM)
print(f"Figuras salvas em {SAVE_DIR}/")
