import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gvae import GVAE_CV

def gvae_cv_loss(x_hat, x, mu, sigma=1.0):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL divergence with fixed variance
    sigma2 = sigma ** 2
    kld = 0.5 * torch.sum(mu.pow(2) + sigma2 - 1 - torch.log(torch.tensor(sigma2)))
    return (recon_loss + kld) / x.size(0)  # Normalize by batch size

# Training function
def train(gvae_cv, data_loader, epochs=20, device='cpu', sigma=1.0):
    optimizer = Adam(gvae_cv.parameters(), lr=0.001)
    gvae_cv.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.view(-1, 784).to(device)  # Flatten MNIST images to 784 dims
            optimizer.zero_grad()
            
            # Forward pass
            x_hat, mu = gvae_cv(x)
            loss = gvae_cv_loss(x_hat, x, mu, sigma=sigma)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

# Main execution
if __name__ == "__main__":
    # Device setup (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST dataset
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(mnist, batch_size=128, shuffle=True)
    
    # Initialize GVAE-CV
    sigma_value = 0.1  # Tune this in [0.025, 1.0] as per paper
    gvae_cv = GVAE_CV(input_dim=784, hidden_dim=512, latent_dim=20, sigma=sigma_value).to(device)
    
    # Train the model
    train(gvae_cv, data_loader, epochs=20, device=device, sigma=sigma_value)
    
    # Save the model
    torch.save(gvae_cv.state_dict(), "gvae_cv_mnist.pt")
    print("Model saved to gvae_cv_mnist.pt")