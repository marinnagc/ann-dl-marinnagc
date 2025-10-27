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
DATASET = 'mnist'  # 'fashion'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './vae_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

# ======= DATA =======
transform = transforms.ToTensor()
ds = datasets.MNIST if DATASET == 'mnist' else datasets.FashionMNIST
dataset = ds(root='./data', train=True, download=True, transform=transform)
val_size = int(0.1 * len(dataset))
train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

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
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ======= TRAIN =======
losses = []
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, _ in train_loader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    losses.append(train_loss / len(train_loader.dataset))
    print(f'Epoch {epoch+1} | Loss: {train_loss:.2f}')

    # Salvar imagens
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(val_loader))
        x = x.to(DEVICE)
        recon, _, _ = model(x)
        recon = recon.view(-1, 1, 28, 28)
        comparison = torch.cat([x[:8], recon[:8]])
        utils.save_image(comparison.cpu(), f"{SAVE_DIR}/recon_epoch_{epoch+1}.png", nrow=8)

        # Amostragem do espaço latente
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        sample = model.decode(sample).view(-1, 1, 28, 28)
        utils.save_image(sample.cpu(), f"{SAVE_DIR}/sample_epoch_{epoch+1}.png", nrow=8)

# ======= LOSS PLOT =======
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{SAVE_DIR}/loss_curve.png")
plt.close()

# ======= LATENT SPACE VISUALIZATION =======
if LATENT_DIM > 3:
    print("Reducing with TSNE...")
    z_list, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            mu, _ = model.encode(x.view(-1, 28*28))
            z_list.append(mu.cpu())
            labels.append(y)
    z_all = torch.cat(z_list).numpy()
    y_all = torch.cat(labels).numpy()
    z_tsne = TSNE(n_components=2).fit_transform(z_all)
    plt.figure(figsize=(6,6))
    plt.scatter(z_tsne[:,0], z_tsne[:,1], c=y_all, cmap='tab10', s=5)
    plt.colorbar()
    plt.savefig(f"{SAVE_DIR}/latent_tsne.png")
    plt.close()
