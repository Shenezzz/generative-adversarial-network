import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from models.discriminator import Discriminator
from models.generator import Generator

from data import get_mnist_data

from train import train_gan


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(42)

    NOISE_DIM = 100  # length of the random noise vector z fed to the generator
    EPOCHS = 30
    
    train_loader = get_mnist_data()
    
    generator = Generator(NOISE_DIM).to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Loss function and optimizer
    # The ppaer uses binary cross entropy for the value function V(G,D).
    # log(D(x)) and log(1 - D(G(x)))
    # These are essentially the two parts of BCE
    criterion = nn.BCEWithLogitsLoss()

    # We set seperate optimizer for each network.
    # both never upadated at the same time.
    # seperate alternating gradient steps as shown in the paper
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    d_losses, g_losses = train_gan(EPOCHS,train_loader,device,criterion,NOISE_DIM,optimizer_D,optimizer_G,discriminator,generator)

    save_path = "gan_loss_curves.png"
    plt.figure(figsize=(8, 5))
    plt.plot(d_losses, label="Discriminator loss")
    plt.plot(g_losses, label="Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "GAN Training Losses (adversarial dynamic -- expect oscillation, not smooth decay)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/"+save_path, dpi=150)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
