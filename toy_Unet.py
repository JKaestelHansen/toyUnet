# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from torch.utils.data import Dataset
from PIL import Image
import os

from torch.utils.data import DataLoader
import torch.optim as optim


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def create_noisy_image_with_random_square(size=128, square_size=32, noise_level=0.1):
    # Create a smaller image with a square in the center
    small_image_size = square_size + 20  # Add some padding for rotation
    small_image = np.zeros((small_image_size, small_image_size))
    start = (small_image_size - square_size) // 2
    end = start + square_size
    small_image[start:end, start:end] = 1  # Fill in the square with ones

    # Rotate the small image at a random angle
    angle = np.random.uniform(0, 360)
    rotated_small_image = rotate(small_image, angle, reshape=False)

    # Create the larger background image
    background_image = np.zeros((size, size))

    # Paste the rotated square onto the background image at random coordinates
    x_offset = np.random.randint(0, size - small_image_size)
    y_offset = np.random.randint(0, size - small_image_size)
    background_image[y_offset:y_offset+small_image_size, x_offset:x_offset+small_image_size] += rotated_small_image

    # Ensure values are still binary (0 or 1) after rotation's interpolation
    background_image = np.clip(background_image, 0, 1)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, background_image.shape)
    noisy_image = background_image + noise

    return noisy_image, background_image

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 2)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64//32, 128//32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128//32, 256//32))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(256//32, 128//32, kernel_size=1))
        self.conv1 = DoubleConv(256//32, 128//32)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(128//32, 64//32, kernel_size=1))
        self.conv2 = DoubleConv(128//32, 64//32)
        self.outc = nn.Conv2d(64//32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = self.conv1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x1], dim=1))
        logits = self.outc(x)
        return logits

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        images, true_masks = batch
        images = images.to(device)
        true_masks = true_masks.to(device)

        optimizer.zero_grad()
        masks_pred = model(images)
        loss = criterion(masks_pred, true_masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images, true_masks = batch
            images = images.to(device)
            true_masks = true_masks.to(device)

            masks_pred = model(images)
            loss = criterion(masks_pred, true_masks)

            running_loss += loss.item()
    return running_loss / len(loader)


def visualize_predictions(images, true_masks, pred_masks):
    plt.figure(figsize=(10, 5 * len(images)))
    
    for idx, (image, true_mask, pred_mask) in enumerate(zip(images, true_masks, pred_masks)):
        plt.subplot(len(images), 3, idx * 3 + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title('Input Image')
        plt.axis('off')
        
        plt.subplot(len(images), 3, idx * 3 + 2)
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        plt.subplot(len(images), 3, idx * 3 + 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.show()

# Example usage after a prediction step in the evaluation loop
# Assuming 'images' and 'true_masks' are from a batch in your dataloader
# pred_masks = torch.sigmoid(masks_pred) > 0.5  # Applying threshold to get binary mask
# visualize_predictions(images.cpu().numpy(), true_masks.cpu().numpy(), pred_masks.cpu().numpy())


# Generate and display the image
noisy_image, background_image = create_noisy_image_with_random_square()
fig, ax = plt.subplots(1,2,figsize=(10, 5))
ax[0].imshow(noisy_image, cmap='gray')
ax[0].set_title('Noisy Random Square')
ax[0].axis('off')  # Hide axis ticks and labels
ax[1].imshow(background_image, cmap='gray')
ax[1].set_title('GT Random Square')
ax[1].axis('off')  # Hide axis ticks and labels
plt.show()


# %%

# CustomDataset class to simply take image and GT numpy as input and make suitable for Unet

class CustomDataset(Dataset):
    def __init__(self, D, GT):
        self.D = torch.tensor(D, dtype=torch.float32).unsqueeze(1)
        self.GT = torch.tensor(GT, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        return self.D[idx], self.GT[idx]

N = 100
D, GT = [], []
for i in range(N):
    noisy_image, background_image = create_noisy_image_with_random_square()
    D.append(noisy_image)
    GT.append(background_image)

# random split indices
train_indices = np.random.choice(N, int(N*0.8), replace=False)
val_indices = np.setdiff1d(np.arange(N), train_indices)

D_train, GT_train = np.array(D)[train_indices], np.array(GT)[train_indices]
D_val, GT_val = np.array(D)[val_indices], np.array(GT)[val_indices]

train_dataset = CustomDataset(D_train, GT_train)
val_dataset = CustomDataset(D_val, GT_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=1, n_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")


# %%
    
    
pred = model(torch.tensor(D_val, dtype=torch.float32).unsqueeze(1))
pred = torch.sigmoid(pred) > 0.5

# show iamges with pred overlay as mask with 0 as transparent
my_cmap = plt.cm.get_cmap('jet')
my_cmap.set_under('w', alpha=0)

# add all predictions of D_val in a grid as subplot
fig, ax = plt.subplots(len(D_val)//5, 3, figsize=(5, 10))
for i in range(len(D_val)//5):
    ax[i,0].imshow(D_val[i], cmap='gray')
    ax[i,0].axis('off')
    ax[i,1].imshow(D_val[i], cmap='gray')
    ax[i,1].imshow(pred[i].squeeze(), cmap=my_cmap, vmin=0.1, alpha=0.5)
    ax[i,1].axis('off')
    ax[i,2].imshow(D_val[i]>0.5, cmap='gray')
    ax[i,2].axis('off')
plt.tight_layout()
    
# %%
