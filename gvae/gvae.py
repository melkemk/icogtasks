import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Encoder: Maps image x to latent Gaussian params (mu, logvar)
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 784 -> 512 
        self.fc2 = nn.Linear(hidden_dim, 256)        # 512 -> 256 
        self.fc_mu = nn.Linear(256, latent_dim)      # 256 -> 20 (mean) 
        self.fc_logvar = nn.Linear(256, latent_dim)  # 256 -> 20 (log variance) 

    def forward(self, x):
        h1 = F.relu(self.fc1(x))      # First hidden layer
        h2 = F.relu(self.fc2(h1))     # Second hidden layer
        mu = self.fc_mu(h2)           # Mean of latent Gaussian
        logvar = self.fc_logvar(h2)   # Log variance of latent Gaussian
        return mu, logvar

# Decoder: Maps latent code z back to image x_hat
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=512, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)        # 20 -> 256
        self.fc2 = nn.Linear(256, hidden_dim)        # 256 -> 512
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 512 -> 784

    def forward(self, z):
        h1 = F.relu(self.fc1(z))      # First hidden layer
        h2 = F.relu(self.fc2(h1))     # Second hidden layer
        x_hat = torch.sigmoid(self.fc3(h2))  # Output image (0-1 via sigmoid)
        return x_hat

# VAE: Combines Encoder and Decoder, handles sampling and forward pass
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # σ = exp(logvar/2)
        eps = torch.randn_like(std)    # Random noise ε ~ N(0, 1)
        return mu + std * eps           # z = μ + σ * ε (reparameterization)

    def forward(self, x):
        mu, logvar = self.encoder(x)         # Encode to Gaussian params
        z = self.reparameterize(mu, logvar)  # Sample latent z
        x_hat = self.decoder(z)              # Decode to reconstructed image
        return x_hat, mu, logvar              # Return all for loss computation
