import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Autoencoder model: compresses and reconstructs data
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Prepare the MNIST data (flatten images to 784-long vectors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model setup
model = Autoencoder()
criterion = nn.MSELoss()  # Measure reconstruction error
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Optimize weights

# Train the autoencoder
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, _ = batch
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Test the autoencoder
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, _ = batch
        outputs = model(inputs)
        test_loss += criterion(outputs, inputs).item()
print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# Visualize original and reconstructed images
test_batch = next(iter(test_loader))
original_images, _ = test_batch
reconstructed_images = model(original_images).detach()

n = 10
plt.figure(figsize=(10, 4))
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(original_images[i].view(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i].view(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Top: Original | Bottom: Reconstructed")
plt.show()
