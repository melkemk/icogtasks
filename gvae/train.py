
from gvae import VAE
import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch.optim import Adam # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore


# Loss function: Reconstruction (BCE) + KL Divergence
def vae_loss(x_hat, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL divergence to enforce N(0, 1) on latent z
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# Training function
def train(vae, data_loader, epochs=20, device='cpu'):
    optimizer = Adam(vae.parameters(), lr=0.001)
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.view(-1, 784).to(device)  # Flatten MNIST images to 784 dims
            
            # Forward pass
            x_hat, mu, logvar = vae(x)
            loss = vae_loss(x_hat, x, mu, logvar)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            optimizer.zero_grad()
        
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
    
    # Initialize VAE
    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=20).to(device)
    
    # Train the model
    train(vae, data_loader, epochs=20, device=device)
    
    # Optional: Save the model
    torch.save(vae.state_dict(), "gvae_mnist.pt")
    print("Model saved to gvae_mnist.pt")