import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture  # For GMM
import numpy as np
import matplotlib.pyplot as plt

# Load the saved VAE model (assuming gvae.py defines the VAE class)
from gvae import VAE  

# Device setup (GPU if available)
device = "cpu" 

def load_data():
    transform = transforms.ToTensor() 
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return test_loader

def load_model():
    vae = VAE(input_dim=784, hidden_dim=512, latent_dim=20).to(device)
    vae.load_state_dict(torch.load("gvae_mnist.pt", map_location=device))
    vae.eval()  # Set the model to evaluation mode
    return vae

def extract_latent_representations(vae, test_loader):
    latent_representations = []
    labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)  # Flatten images to 784 dims
            mu, _ = vae.encoder(x)          # Get latent representations (mu)
            latent_representations.append(mu.cpu().numpy())
            labels.append(y.cpu().numpy())
    X_test = np.vstack(latent_representations)
    y_test = np.hstack(labels)
    return X_test, y_test

def train_gmm(X_test, n_components=10):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_test)
    return gmm

def train_logistic_regression(X_test, y_test):
    classifier = LogisticRegression(max_iter=100)
    classifier.fit(X_test, y_test)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_percentage = 100 * (1 - accuracy)
    return error_percentage

def calculate_bce_loss(vae, test_loader):
    total_bce = 0
    num_samples = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).view(-1, 784)  # Flatten images 
            x_bin = (x > 0.5).float()      # Binarize original images
            mu, _ = vae.encoder(x_bin)     # Encode
            x_hat = vae.decoder(mu)        # Decode (reconstructed images)
            bce = F.binary_cross_entropy(x_hat, x_bin, reduction='sum')
            total_bce += bce.item()
            num_samples += x_bin.size(0)
    avg_bce = total_bce / (num_samples * 784)  # Normalize by number of pixels
    return avg_bce

def monte_carlo_log_likelihood(gmm, vae, data_loader, n_samples=5000):
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32).to(device)
    log_p_z = gmm.score_samples(gmm_samples)
    log_p_x_given_z = np.zeros(n_samples)
    
    with torch.no_grad():
        recon_xs = []
        for i in range(0, n_samples, data_loader.batch_size):
            batch_z = z_samples[i:i + data_loader.batch_size]
            recon_x = vae.decoder(batch_z)
            recon_xs.append(recon_x)
        recon_xs = torch.cat(recon_xs, dim=0)
        
        total_batches = 0
        for x, _ in data_loader:
            x_bin = (x.to(device).view(-1, 784) > 0.5).float()
            batch_size = x_bin.size(0)
            for i in range(0, n_samples, data_loader.batch_size):
                batch_recon = recon_xs[i:i + data_loader.batch_size]
                effective_size = min(batch_recon.size(0), batch_size)
                bce = F.binary_cross_entropy(
                    batch_recon[:effective_size], 
                    x_bin[:effective_size], 
                    reduction='none'
                )
                log_p_x_given_z[i:i + effective_size] += -bce.sum(dim=1).cpu().numpy()
            total_batches += 1
        log_p_x_given_z /= total_batches
    
    log_likelihood = np.mean(log_p_z + log_p_x_given_z)
    return log_likelihood

def calculate_mse_on_masked_region(vae, test_loader):
    total_mse = 0
    total_masked_elements = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            original_x = x.clone().view(-1, 784)  # Save original, flatten
            x = (x > 0.5).float()                # Binarize
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, :14] = 0                     # Mask left half (14 columns, flattened)
            x = x * mask.float()                 # Apply mask
            x = (x > 0.5).float().view(-1, 784)  # Re-binarize and flatten
            mu, _ = vae.encoder(x)               # Encode
            x_hat = vae.decoder(mu)              # Decode
            mask_flat = mask.view(-1, 784)
            mse = F.mse_loss(x_hat[~mask_flat], original_x[~mask_flat], reduction='sum')
            total_mse += mse.item()
            total_masked_elements += (~mask_flat).sum().item()
    avg_mse = total_mse / total_masked_elements
    return avg_mse

def main():
    test_loader = load_data()
    vae = load_model()
    X_test, y_test = extract_latent_representations(vae, test_loader)
    gmm = train_gmm(X_test)
    error_percentage = train_logistic_regression(X_test, y_test)
    avg_bce = calculate_bce_loss(vae, test_loader)
    avg_mse = calculate_mse_on_masked_region(vae, test_loader)
    
    print(f"Shape of latent representations (X_test): {X_test.shape}")
    print(f"Shape of labels (y_test): {y_test.shape}")
    print(f"GMM trained with {gmm.n_components} components.")
    print(f"Classification Error Percentage: {error_percentage:.2f}%")
    print(f"Average BCE Loss per pixel: {avg_bce:.4f}")
    print(f"Avg MSE on masked region: {avg_mse:.2f}")

if __name__ == "__main__":
    main()

test_loader = load_data()
vae = load_model()

def bce_loss(model, loader):
        model.eval()
        total_bce = 0.0
        total_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, _ in loader:
                data = data.view(data.size(0), -1)
                data = (data > 0.5).float()
                recon_data, _, _ = model(data)
                recon_data = recon_data.view(data.size(0), -1)
                
                # Calculate BCE\n",
                #bce = vae_loss_eval(recon_data, data, mu, logvar)
                #total_bce += bce.item() 
                
                bce = F.binary_cross_entropy(recon_data, data, reduction='sum')
                total_bce += bce.item() 
                
                # Mask exactly half of the image columns\n",
                mask = torch.ones_like(data, dtype=torch.bool)
                mask[:, : data.size(1) // 2] = 0
                masked_data = data * mask.float()
                
                # Calculate MSE over the masked (hidden) parts\n",
                mse = F.mse_loss(recon_data[~mask], data[~mask], reduction='sum')
                total_mse += mse.item()
                total_samples += data.size(0)
    
        # Normalize by the total number of elements
        avg_bce = total_bce / total_samples  
        avg_mse = total_mse / (total_samples * data.size(1) // 2)
        print(f"BCE: {avg_bce:.4f}, Masked MSE: {avg_mse:.4f}")
        return avg_bce, avg_mse
    
    # Evaluate the model using the calculate_error function
bce_loss(vae, test_loader)

 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_space(vae, test_loader, save_path="latent_space.png"):
    vae.eval()
    latent_representations = []
    labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(-1, 784).to(device)  # Flatten images
            mu, _ = vae.encoder(x)          # Get latent representations
            latent_representations.append(mu.cpu().numpy())
            labels.append(y.cpu().numpy())
    
    X_test = np.vstack(latent_representations)
    y_test = np.hstack(labels)
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_test)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label="Digit Labels")
    plt.title("Latent Space Visualization using t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    # Save the figure
    plt.savefig(save_path)
    plt.show()

# Call the function to visualize
visualize_latent_space(vae, test_loader)