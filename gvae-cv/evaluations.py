import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Assuming GVAE_CV is defined in gvae.py with the same structure as before
from gvae import GVAE_CV

# Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, sigma_value):
    gvae_cv = GVAE_CV(input_dim=784, hidden_dim=512, latent_dim=20, sigma=sigma_value).to(device)
    gvae_cv.load_state_dict(torch.load(model_path, map_location=device))
    gvae_cv.eval()
    return gvae_cv

def extract_latent_representations(model, data_loader):
    latent_representations = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(-1, 784).to(device)
            mu = model.encoder(x)
            latent_representations.append(mu.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.vstack(latent_representations), np.hstack(labels)

def calculate_bce_loss(model, data_loader):
    total_bce = 0
    num_samples = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device).view(-1, 784)
            x_bin = (x > 0.5).float()
            mu = model.encoder(x_bin)
            x_hat = model.decoder(mu)
            bce = F.binary_cross_entropy(x_hat, x_bin, reduction='sum')
            total_bce += bce.item()
            num_samples += x_bin.size(0)
    return total_bce / (num_samples * 784)

def monte_carlo_log_likelihood(gmm, model, data_loader, n_samples=5000):
    gmm_samples, _ = gmm.sample(n_samples)
    z_samples = torch.tensor(gmm_samples, dtype=torch.float32).to(device)
    log_p_z = gmm.score_samples(gmm_samples)
    log_p_x_given_z = np.zeros(n_samples)
    
    with torch.no_grad():
        # Precompute reconstructions
        recon_xs = []
        for i in range(0, n_samples, data_loader.batch_size):
            batch_z = z_samples[i:i + data_loader.batch_size]
            recon_x = model.decoder(batch_z)
            recon_xs.append(recon_x)
        recon_xs = torch.cat(recon_xs, dim=0)
        
        # Average p(x|z) over all test data
        total_batches = 0
        for x, _ in data_loader:
            x_bin = (x.to(device).view(-1, 784) > 0.5).float()
            batch_size = x_bin.size(0)  # Actual batch size (may be < 128 for last batch)
            for i in range(0, n_samples, data_loader.batch_size):
                batch_recon = recon_xs[i:i + data_loader.batch_size]
                # Ensure sizes match by slicing batch_recon to match x_bin
                effective_size = min(batch_recon.size(0), batch_size)
                bce = F.binary_cross_entropy(batch_recon[:effective_size], x_bin[:effective_size], reduction='none')
                log_p_x_given_z[i:i + effective_size] += -bce.sum(dim=1).cpu().numpy()
            total_batches += 1
        log_p_x_given_z /= total_batches
    
    return np.mean(log_p_z + log_p_x_given_z)

def calculate_mse_on_masked_region(model, data_loader):
    total_mse = 0
    total_masked_elements = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            original_x = x.clone().view(-1, 784)
            x = (x > 0.5).float()
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, :14] = 0
            x = x * mask.float()
            x = (x > 0.5).float().view(-1, 784)
            mu = model.encoder(x)
            x_hat = model.decoder(mu)
            mask_flat = mask.view(-1, 784)
            mse = F.mse_loss(x_hat[~mask_flat], original_x[~mask_flat], reduction='sum')
            total_mse += mse.item()
            total_masked_elements += (~mask_flat).sum().item()
    return total_mse / total_masked_elements

# Main execution
if __name__ == "__main__":
    # Load MNIST test dataset
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize GVAE-CV and load the saved model
    sigma_value = 0.1  # Match the sigma used during training
    model_path = "gvae_cv_mnist.pt"
    gvae_cv = load_model(model_path, sigma_value)

    # Extract latent representations and labels
    X_test, y_test = extract_latent_representations(gvae_cv, test_loader)
    print(f"Shape of latent representations (X_test): {X_test.shape}")
    print(f"Shape of labels (y_test): {y_test.shape}")

    # Train a GMM on the latent representations
    n_components = 10
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_test)
    print(f"GMM trained with {n_components} components.")

    # Logistic Regression Classifier
    classifier = LogisticRegression(max_iter=100)
    classifier.fit(X_test, y_test)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_percentage = 100 * (1 - accuracy)
    print(f"Classification Error Percentage: {error_percentage:.2f}%")

    # Calculate BCE Loss
    avg_bce = calculate_bce_loss(gvae_cv, test_loader)
    print(f"Average BCE Loss per pixel: {avg_bce:.4f}")

    # Compute Monte Carlo Log Likelihood
    log_likelihood = monte_carlo_log_likelihood(gmm, gvae_cv, test_loader, n_samples=5000)
    print(f"Monte Carlo Log Likelihood: {log_likelihood:.2f}")

    # MSE on Masked Region
    avg_mse = calculate_mse_on_masked_region(gvae_cv, test_loader)
    print(f"Avg MSE on masked region: {avg_mse:.2f}")