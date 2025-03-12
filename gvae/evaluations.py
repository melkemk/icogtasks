import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the saved VAE model
from gvae import VAE  

# Device setup (GPU if available)
device = "cpu"

# Load MNIST test dataset
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize VAE and load the saved model
vae = VAE(input_dim=784, hidden_dim=512, latent_dim=20).to(device)
vae.load_state_dict(torch.load("gvae_mnist.pt", map_location=device))
vae.eval()  # Set the model to evaluation fmode
 
# Extract latent representations (mu) and labels from the test set
latent_representations = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.view(-1, 784).to(device)  # Flatten images to 784 dims
        mu, _ = vae.encoder(x)          # Get latent representations (mu)
        latent_representations.append(mu.cpu().numpy())
        labels.append(y.cpu().numpy())

# Stack all latent representations and labels into single arrays
X_test = np.vstack(latent_representations)
y_test = np.hstack(labels)

print(f"Shape of latent representations (X_test): {X_test.shape}")
print(f"Shape of labels (y_test): {y_test.shape}")

classifier = LogisticRegression(max_iter=100)
classifier.fit(X_test, y_test)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error_percentage = 100 * (1 - accuracy)

print(f"Classification Error Percentage: {error_percentage:.2f}%")



total_mse = 0
total_masked_elements = 0

with torch.no_grad():

    for x, _ in test_loader:
        x = x.to(device)
        original_x = x.clone().view(-1, 784)  # Save original, flatten
        x = (x > 0.5).float()                # Binarize
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, :, :, :14] = 0                # Mask left half (14 columns)
        x = x * mask.float()                  # Apply mask
        x = (x > 0.5).float().view(-1, 784)  # Re-binarize and flatten
        mu, _ = vae.encoder(x)                # Encode
        x_hat = vae.decoder(mu)               # Decode
        mask_flat = mask.view(-1, 784)
        mse = F.mse_loss(x_hat[~mask_flat], original_x[~mask_flat], reduction='sum')
        total_mse += mse.item()
        total_masked_elements += (~mask_flat).sum().item()

print(f"Avg MSE on masked region: {total_mse / total_masked_elements:.2f}")