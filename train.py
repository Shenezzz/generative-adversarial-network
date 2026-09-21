import torch
import matplotlib.pyplot as plt
import numpy as np


def train_gan(
    epochs,
    train_loader,
    device,
    criterion,
    NOISE_DIM,
    optimizer_D,
    optimizer_G,
    discriminator,
    generator,
):
    fixed_noise = torch.randn(64, NOISE_DIM, device=device)

    d_losses, g_losses = [], []

    for epoch in range(1, epochs + 1):
        epoch_d_loss, epoch_g_loss = 0.0, 0.0

        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            # Ground-truth targets: real images are labeled 1, fakes are labeled 0.
            real_labels = torch.full((current_batch_size, 1),0.9, device=device) # to prevent the discriminator from being overconfident
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            #  (1) Train the Discriminator
            #      maximize log D(x) + log(1 - D(G(z)))
            #      <=> minimize BCE(D(x), 1) + BCE(D(G(z)), 0)

            optimizer_D.zero_grad()

            # real images should be classified as real (label 1)
            real_preds = discriminator(real_images)
            loss_d_real = criterion(real_preds, real_labels)

            # generate fake images, should be classified as fake (label 0)
            z = torch.randn(current_batch_size, NOISE_DIM, device=device)
            fake_images = generator(z)

            # .detach() is important here: it stops gradients from flowing
            # back into the generator during the DISCRIMINATOR's update --
            # we only want to update D's weights at this step, not G's.
            fake_preds = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_preds, fake_labels)

            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            optimizer_D.step()

            #  (2) Train the Generator
            #      Textbook version: minimize log(1 - D(G(z)))
            #      NON-SATURATING version: maximize log D(G(z)) instead.
            #      implemented here simply by training G to make the
            #      discriminator output REAL (label 1) for its fakes.
            #      This is mathematically the non-saturating trick: we
            #      compute BCE(D(G(z)), 1) instead of BCE(D(G(z)), 0),
            #      which gives a much stronger gradient early in training.

            optimizer_G.zero_grad()

            # Forward the SAME fake images through D again, but WITHOUT
            # .detach() this time -- we need gradients to flow all the
            # way back into the generator's weights now.
            fake_preds_for_g = discriminator(fake_images)
            loss_G = criterion(
                fake_preds_for_g, real_labels
            )  # note: real_labels, not fake!
            loss_G.backward()
            optimizer_G.step()

            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(
            f"Epoch {epoch:2d}/{epochs} | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f}"
        )


        # Save a snapshot of generated images every few epochs, so we
        # can watch quality improve over the course of training.
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            save_generated_grid(generator, fixed_noise, f"gan_epoch_{epoch:03d}.png")
            torch.save({"generator": generator.state_dict(),"discriminator":discriminator.state_dict()},"checkpoint/model_state.pt")

    return d_losses, g_losses


def save_generated_grid(generator, noise, save_path, grid_size=8):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    # Undo the [-1,1] normalization back to [0,1] for display.
    fake_images = (fake_images + 1) / 2
    fake_images = fake_images.clamp(0, 1)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i in range(grid_size * grid_size):
        axes[i // grid_size, i % grid_size].imshow(
            fake_images[i, 0].numpy(), cmap="gray"
        )
        axes[i // grid_size, i % grid_size].axis("off")
    plt.suptitle(f"Generated digits — {save_path}")
    plt.tight_layout()
    plt.savefig("figures/"+save_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {save_path}")
    generator.train()
