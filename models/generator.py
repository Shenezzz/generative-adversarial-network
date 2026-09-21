import torch
import torch.nn as nn

# The generator G(x)
# Takes a random noise vector z (NOISE_DIM numbers) and transforms
# it, step by step, into a full 28x28 image. This is essentially the
# DECODER half of an autoencoder — same
# idea of "expand a small vector into a full image" — the difference
# is what TRAINS it: no reconstruction target here, only the
# discriminator's feedback.
class Generator(nn.Module):
    def __init__(self,noise_dim):
        super().__init__()
        
        self.fc = nn.Linear(noise_dim,128*7*7)
        self.bn_fc = nn.BatchNorm1d(128*7*7)
        
        self.deconv = nn.Sequential(
            # (B, 128, 7, 7) -> (B, 64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (B, 64, 14, 14) -> (B, 1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # squashes output to [-1, 1], matching our normalized data range
        ) 
    
    def forward(self,z):
        h = torch.relu(self.bn_fc(self.fc(z)))       
        h = h.view(h.size(0),128,7,7)
        img = self.deconv(h)
        return img