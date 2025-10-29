import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# =============== CONFIG ===============
BATCH_SIZE = 128
EPOCHS = 20
LATENT_DIM = 16           # pode mudar depois p/ testar qualidade vs separação
LR = 1e-3
DATASET = 'mnist'         # ou 'fashion'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './vae_outputs_cnn_with_tsne_and_val'
os.makedirs(SAVE_DIR, exist_ok=True)

# =============== DATA ===============
transform = transforms.Compose([
    transforms.ToTensor()  # MNIST já vem [0,1], então ok
])

ds = datasets.MNIST if DATASET == 'mnist' else datasets.FashionMNIST
full_ds = ds(root='./data', train=True, download=True, transform=transform)

val_size = int(0.1 * len(full_ds))
train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# =============== MODEL ===============
class CNN_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # ----- Encoder -----
        # in: [B,1,28,28]
        self.enc_conv1 = nn.Conv2d(1, 32, 4, 2, 1)   # -> [B,32,14,14]
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)  # -> [B,64,7,7]

        self.enc_fc_mu     = nn.Linear(64*7*7, LATENT_DIM)
        self.enc_fc_logvar = nn.Linear(64*7*7, LATENT_DIM)

        # ----- Decoder -----
        self.dec_fc = nn.Linear(LATENT_DIM, 64*7*7)
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # [7,7] -> [14,14]
        self.dec_deconv2 = nn.ConvTranspose2d(32, 1, 4, 2, 1)   # [14,14] -> [28,28]

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)           # flatten [B,64*7*7]
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, 64, 7, 7)            # unflatten
        x = F.relu(self.dec_deconv1(x))
        x = torch.sigmoid(self.dec_deconv2(x))  # final [B,1,28,28]
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
    # recon_x e x estão [B,1,28,28]
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# =============== TREINO + VALIDAÇÃO ===============
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # ----- TREINO -----
    model.train()
    running_train_loss = 0.0

    for x, _ in train_loader:
        x = x.to(DEVICE)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward() # Backpropagation, cálculo do gradiente, atualização dos pesos
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # ----- VALIDAÇÃO (AVALIAÇÃO) -----
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

    # ----- VISUALIZAÇÕES DA ÉPOCA -----
    with torch.no_grad(): # Desativa o cálculo de gradientes para economizar memória e acelerar a inferência durante a validação
        # comparação entrada vs reconstrução
        x_val, _ = next(iter(val_loader))
        x_val = x_val.to(DEVICE)
        recon_val, _, _ = model(x_val)
        comparison = torch.cat([x_val[:8], recon_val[:8]])  # 8 originais + 8 reconstruídas
        utils.save_image(
            comparison.cpu(),
            f"{SAVE_DIR}/recon_epoch_{epoch+1}.png",
            nrow=8
        )

        # amostras novas puras do prior N(0,I)
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        sample = model.decode(sample)
        utils.save_image(
            sample.cpu(),
            f"{SAVE_DIR}/sample_epoch_{epoch+1}.png",
            nrow=8
        )

# =============== LOSS CURVE (train vs val) ===============
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title("CNN-VAE Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/loss_curve.png")
plt.close()

# =============== LATENT SPACE VIS (t-SNE) ===============
print("Plotando espaço latente com t-SNE...")
model.eval()
z_list, y_list = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)
        mu, _ = model.encode(x)
        z_list.append(mu.cpu())
        y_list.append(y)

z_all = torch.cat(z_list).numpy()   # [N, LATENT_DIM]
y_all = torch.cat(y_list).numpy()   # [N]

z_tsne = TSNE(n_components=2).fit_transform(z_all)

plt.figure(figsize=(6,6))
plt.scatter(z_tsne[:,0], z_tsne[:,1], c=y_all, cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE do Espaço Latente (μ)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/latent_tsne.png")
plt.close()

print(f"Tudo salvo em {SAVE_DIR}/")
