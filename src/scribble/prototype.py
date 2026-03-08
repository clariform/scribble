import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# -----------------------
# Hyperparameters / setup
# -----------------------
latent_dim = 128
batch_size = 128
lr = 2e-4
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # maps [0,1] -> [-1,1]
])

datasets_root = os.environ.get("WHISK_ML_DATASETS")
if not datasets_root:
    raise RuntimeError("Environment variable WHISK_ML_DATASETS is not set.")

# MNIST will create/use a subfolder named 'MNIST' under this root
data_root = os.path.join(datasets_root, "mnist")

dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,  # keeps batch size consistent (helps BatchNorm)
    num_workers=2,   # adjust if needed
    pin_memory=(device.type == "cuda"),
)

# -----------------------
# Outputs (headless-friendly)
# -----------------------
out_dir = os.path.join(data_root, "outputs")
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# Models
# -----------------------
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 784),
            nn.Tanh(),  # output in [-1,1] to match Normalize(0.5,0.5)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        x_flat = self.net(z)               # (B, 784)
        x_img = x_flat.view(-1, 1, 28, 28) # (B, 1, 28, 28)
        return x_img


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),  # outputs probability in [0,1]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 1, 28, 28) or (B, 784)
        img_flat = img.view(img.size(0), -1)  # (B, 784)
        return self.net(img_flat)             # (B, 1)


# -----------------------
# Instantiate + Optimizers
# -----------------------
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# -----------------------
# Training loop
# -----------------------
G.train()
D.train()

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        bsz = real_imgs.size(0)

        real_labels = torch.ones(bsz, 1, device=device)
        fake_labels = torch.zeros(bsz, 1, device=device)

        # -------------------------
        # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        # -------------------------
        z = torch.randn(bsz, latent_dim, device=device)
        fake_imgs = G(z)

        d_real = D(real_imgs)
        d_fake = D(fake_imgs.detach())

        real_loss = criterion(d_real, real_labels)
        fake_loss = criterion(d_fake, fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -------------------------
        # Train Generator: maximize log(D(G(z)))  (i.e., fool D)
        # -------------------------
        z = torch.randn(bsz, latent_dim, device=device)
        generated_imgs = G(z)

        d_generated = D(generated_imgs)
        g_loss = criterion(d_generated, real_labels)  # want D(G(z)) -> 1

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if i % 200 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} "
                f"Batch {i}/{len(dataloader)} "
                f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"
            )

    # -------------------------
    # Save samples each epoch (headless)
    # -------------------------
    G.eval()
    with torch.no_grad():
        fake = G(torch.randn(64, latent_dim, device=device))
        save_path = os.path.join(out_dir, f"epoch_{epoch+1:03d}.png")
        save_image(fake, save_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"[saved] {save_path}")
    G.train()
