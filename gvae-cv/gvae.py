import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=480, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=512, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        x_hat = torch.sigmoid(self.fc3(h2))
        return x_hat

class GVAE_CV(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=20, sigma=1.0):
        super(GVAE_CV, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.sigma = sigma

    def reparameterize(self, mu):
        std = torch.ones_like(mu) * self.sigma
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, x):
        mu = self.encoder(x)
        z = self.reparameterize(mu)
        x_hat = self.decoder(z)
        return x_hat, mu

