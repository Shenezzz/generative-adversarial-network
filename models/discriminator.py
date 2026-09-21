import torch
import torch.nn as nn


# Discriminator  D(x)
# Takes an image (real or fake) and outputs a single number in
# (0, 1): the estimated probability that the image is real. This is
# just a standard image classifier — the same conv->BN->activation
# pattern you already know from the CNN scripts, except the final
# output is ONE sigmoid-activated number instead of class scores.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),  
            # LeakyReLU (not ReLU) is standard in GAN
            # discriminators -- keeps a small gradient
            # flowing even for negative activations,
            # which helps training stability.
            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        logit = self.fc(h)
        # prob = torch.sigmoid(logit)  # squash to (0,1) -- "probability this is real"
        return logit
