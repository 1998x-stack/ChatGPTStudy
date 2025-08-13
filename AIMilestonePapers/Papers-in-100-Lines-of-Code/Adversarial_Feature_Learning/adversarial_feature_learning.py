import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

# --- 1. Data Preparation ---
def load_mnist_data(batch_size=128, device='cuda'):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# --- 2. Network Definitions ---
class Generator(nn.Module):
    def __init__(self, latent_dim=50):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 784), nn.Tanh()  # Tanh for [-1,1] range
        )

    def forward(self, z):
        return self.model(z)

class Encoder(nn.Module):
    def __init__(self, latent_dim=50):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, latent_dim), nn.Tanh()  # Tanh for latent space
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784 + 50, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x, z):
        return self.model(torch.cat((x, z), dim=1))

# --- 3. Training Loop ---
def train(generator, encoder, discriminator, dataloader, device='cuda', 
          epochs=100, lr=2e-4, gamma=0.99995, save_interval=10):
    # Optimizers
    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    e_optim = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optim, gamma)
    e_scheduler = optim.lr_scheduler.ExponentialLR(e_optim, gamma)
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optim, gamma)

    # Create save directory
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in tqdm(range(epochs), desc="Training"):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1).to(device)  # Flatten images

            # Sample latent vectors
            z = torch.randn(batch_size, 50, device=device)

            # Generate fake images
            fake_images = generator(z)

            # Encode real images
            encoded_z = encoder(real_images)

            # Discriminator loss
            d_real = discriminator(real_images, encoded_z)
            d_fake = discriminator(fake_images, z)
            d_loss = criterion(d_real, torch.ones(batch_size, 1, device=device)) + \
                     criterion(d_fake, torch.zeros(batch_size, 1, device=device))

            # Update Discriminator
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Generator/Encoder loss
            fake_images = generator(z)
            encoded_z = encoder(real_images)
            g_loss = criterion(discriminator(fake_images, z), torch.ones(batch_size, 1, device=device)) + \
                     criterion(discriminator(real_images, encoded_z), torch.zeros(batch_size, 1, device=device))

            # Update Generator/Encoder
            g_optim.zero_grad()
            e_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            e_optim.step()

            # Logging
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G/E Loss: {g_loss.item():.4f}")

        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_{epoch}.pth")
            torch.save(encoder.state_dict(), f"checkpoints/encoder_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_{epoch}.pth")

        # Step schedulers
        g_scheduler.step()
        e_scheduler.step()
        d_scheduler.step()

# --- 4. Visualization ---
def visualize(generator, encoder, device='cuda'):
    generator.eval()
    encoder.eval()

    # Generate from random latent space
    z = torch.randn(20, 50, device=device)
    fake_images = generator(z).detach().cpu().numpy()

    # Reconstruct from real images
    _, (real_images, _) = next(iter(dataloader))
    real_images = real_images.to(device)
    encoded_z = encoder(real_images)
    recon_images = generator(encoded_z).detach().cpu().numpy()

    # Plot results
    plt.figure(figsize=(18, 3.5))
    for i in range(20):
        plt.subplot(3, 20, 1 + i)
        plt.imshow(fake_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Generated", fontsize=17)

        plt.subplot(3, 20, 21 + i)
        plt.imshow(real_images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Original", fontsize=17)

        plt.subplot(3, 20, 41 + i)
        plt.imshow(recon_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Reconstructed", fontsize=17)

    plt.tight_layout()
    plt.savefig("results/adversarial_feature_learning.png")
    plt.close()

# --- 5. Main Execution ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    dataloader = load_mnist_data(batch_size=128, device=device)

    # Initialize networks
    generator = Generator().to(device)
    encoder = Encoder().to(device)
    discriminator = Discriminator().to(device)

    # Train
    train(generator, encoder, discriminator, dataloader, device=device, epochs=100)

    # Visualize
    visualize(generator, encoder, device=device)