import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os

# =============== CONFIG ===============
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 16
LR = 1e-3
DATASET = 'mnist'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './vae_outputs_cnn'
os.makedirs(SAVE_DIR, exist_ok=True)

# =============== DATA ===============
transform = transforms.Compose([
    transforms.ToTensor()
])

ds = datasets.MNIST if DATASET == 'mnist' else datasets.FashionMNIST
dataset = ds(root='./data', train=True, download=True, transform=transform)
val_size = int(0.1 * len(dataset))
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =============== MODEL ===============
class CNN_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # [28,28] -> [14,14]
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1) # [14,14] -> [7,7]
        self.enc_fc_mu = nn.Linear(64*7*7, LATENT_DIM)
        self.enc_fc_logvar = nn.Linear(64*7*7, LATENT_DIM)

        # Decoder
        self.dec_fc = nn.Linear(LATENT_DIM, 64*7*7)
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # [7,7] -> [14,14]
        self.dec_deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)  # [14,14] -> [28,28]

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.dec_deconv1(x))
        x = torch.sigmoid(self.dec_deconv2(x))  # [B, 1, 28, 28]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

model = CNN_VAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =============== LOSS ===============
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# =============== TRAIN ===============
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

    avg_loss = train_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.2f}")

    # Reconstruções
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(val_loader))
        x = x.to(DEVICE)
        recon, _, _ = model(x)
        comparison = torch.cat([x[:8], recon[:8]])
        utils.save_image(comparison.cpu(), f"{SAVE_DIR}/recon_epoch_{epoch+1}.png", nrow=8)

        # Amostras do espaço latente
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        sample = model.decode(sample)
        utils.save_image(sample.cpu(), f"{SAVE_DIR}/sample_epoch_{epoch+1}.png", nrow=8)

# =============== LOSS PLOT ===============
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{SAVE_DIR}/loss_curve.png")
plt.close()

from sklearn.manifold import TSNE

# ======== LATENT SPACE PLOT (opcional) =========
print("Plotando espaço latente com t-SNE...")
model.eval()
z_list, y_list = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        mu, _ = model.encode(x)
        z_list.append(mu.cpu())
        y_list.append(y)

z_all = torch.cat(z_list).numpy()
y_all = torch.cat(y_list).numpy()

z_tsne = TSNE(n_components=2).fit_transform(z_all)
plt.figure(figsize=(6,6))
plt.scatter(z_tsne[:,0], z_tsne[:,1], c=y_all, cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE do Espaço Latente (μ)")
plt.savefig(f"{SAVE_DIR}/latent_tsne.png")
plt.close()
