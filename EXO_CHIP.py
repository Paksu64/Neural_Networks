# Improved Polyp Segmentation Model
# This notebook addresses issues with segmentation accuracy by implementing:
# 1. Enhanced network architecture with skip connections
# 2. Improved loss functions for class imbalance
# 3. Optimized training parameters
# 4. Better data augmentation
# 5. Post-processing techniques

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
png_image_path = "/kaggle/input/cvcclinicdb/PNG/Original"
png_mask_path = "/kaggle/input/cvcclinicdb/PNG/Ground Truth"

# List image files
image_files = sorted([f for f in os.listdir(png_image_path) if f.lower().endswith(".png")])
mask_files = sorted([f for f in os.listdir(png_mask_path) if f.lower().endswith(".png")])

print("Total image files:", len(image_files))
print("Total mask files:", len(mask_files))

# Split dataset with stratified sampling to ensure balanced distribution
train_image_files, test_image_files, train_mask_files, test_mask_files = train_test_split(
    image_files,
    mask_files,
    test_size=0.2,
    random_state=42  # Fixed seed for reproducibility
)

print("Train image files count:", len(train_image_files))
print("Test image files count:", len(test_image_files))

# Further split test set to create a validation set
test_image_files, val_image_files, test_mask_files, val_mask_files = train_test_split(
    test_image_files,
    test_mask_files,
    test_size=0.5,
    random_state=42
)

print("Test image files count:", len(test_image_files))
print("Validation image files count:", len(val_image_files))

# Enhanced data augmentation for medical imaging
train_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # More controlled brightness/contrast changes for medical images
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
        A.RandomGamma(gamma_limit=(90, 110), p=1),
    ], p=0.3),
    # Careful with elastic transforms - they should preserve important structures
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(distort_limit=0.05, p=0.5),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
    ToTensorV2()
])

# Simpler transforms for validation and test sets
val_test_transforms = A.Compose([
    A.Resize(256, 256),
    ToTensorV2()
])

class PolypDataset(Dataset):
    def __init__(self, image_files, mask_files, png_root, png_mask_root, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.png_root = png_root
        self.png_mask_root = png_mask_root
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = self.mask_files[idx]
    
        img_path = os.path.join(self.png_root, image_file)
        mask_path = os.path.join(self.png_mask_root, mask_file)
    
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
    
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
    
        if not torch.is_floating_point(img):
            img = img.float() / 255.0
    
        if not torch.is_floating_point(mask):
            mask = mask.float() / 255.0
            
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
    
        return img, mask

# Create datasets with the appropriate transforms
train_dataset = PolypDataset(
    train_image_files, train_mask_files,
    png_root=png_image_path, png_mask_root=png_mask_path,
    transform=train_transforms
)

val_dataset = PolypDataset(
    val_image_files, val_mask_files,
    png_root=png_image_path, png_mask_root=png_mask_path,
    transform=val_test_transforms
)

test_dataset = PolypDataset(
    test_image_files, test_mask_files,
    png_root=png_image_path, png_mask_root=png_mask_path,
    transform=val_test_transforms
)

# Smaller batch size to prevent overfitting
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Train dataloader size: {len(train_dataloader)}")
print(f"Validation dataloader size: {len(val_dataloader)}")
print(f"Test dataloader size: {len(test_dataloader)}")

# Check class balance in training data
def check_class_balance(dataloader):
    positive_pixels = 0
    total_pixels = 0
    
    for _, masks in dataloader:
        positive_pixels += (masks > 0.5).sum().item()
        total_pixels += masks.numel()
    
    balance = positive_pixels / total_pixels
    print(f"Positive class ratio: {balance:.4f} ({positive_pixels}/{total_pixels})")
    return balance

class_ratio = check_class_balance(train_dataloader)

# Basic convolutional block with improved normalization and activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # Add dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

# U-Net ahhh
class ImprovedEXOCHIP(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedEXOCHIP, self).__init__()
        # Encoder
        self.conv1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)  # 512 = 256 + 256 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 256 = 128 + 128 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)   # 128 = 64 + 64 (skip connection)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(96, 32)    # 96 = 32 + 64 (skip connection)
        
        # Output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        self.attention3 = AttentionGate(256, 256, 128)
        self.attention2 = AttentionGate(128, 128, 64)
        self.attention1 = AttentionGate(64, 64, 32)
        
        # Refinement module with dilated convolutions for better context capturing
        self.refinement = nn.Sequential(
            nn.Conv2d(num_classes, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2),  # Dilated convolution
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.conv4(p3)
        p4 = self.pool4(x4)
        
        # Apply transformer to bottleneck features
        B, C, H, W = p4.shape
        p4_flat = p4.view(B, C, -1).permute(0, 2, 1)
        trans_out = self.transformer_encoder(p4_flat)
        trans_out = trans_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Decoder path with skip connections
        up4 = self.up4(trans_out)
        x3_att = self.attention3(x3, up4)
        merge4 = torch.cat([up4, x3_att], dim=1)
        dec4 = self.dec4(merge4)
        
        up3 = self.up3(dec4)
        x2_att = self.attention2(x2, up3)
        merge3 = torch.cat([up3, x2_att], dim=1)
        dec3 = self.dec3(merge3)
        
        up2 = self.up2(dec3)
        x1_att = self.attention1(x1, up2)
        merge2 = torch.cat([up2, x1_att], dim=1)
        dec2 = self.dec2(merge2)
        
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, x1], dim=1)
        dec1 = self.dec1(merge1)
        
        seg = self.final_conv(dec1)
        refined = self.refinement(seg)
        
        return refined

# Attention gate for better feature selection
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

# Improved DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

# Focal Loss to address class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Focal weights
        pt = probs_flat * targets_flat + (1 - probs_flat) * (1 - targets_flat)
        focal_weight = (1 - pt) ** self.gamma
        
        # Binary cross entropy
        bce = -self.alpha * targets_flat * torch.log(probs_flat + 1e-7) - \
              (1 - self.alpha) * (1 - targets_flat) * torch.log(1 - probs_flat + 1e-7)
        
        focal_loss = focal_weight * bce
        return focal_loss.mean()

# Combined loss function with weighting based on class imbalance
class CombinedLoss(nn.Module):
    def __init__(self, class_ratio=0.1, weight_dice=0.5, weight_focal=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=1-class_ratio)  # Alpha weighted for class imbalance
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        
    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        return self.weight_dice * dice_loss + self.weight_focal * focal_loss

def post_process_mask(pred_mask, min_size=100):
    """Clean up the prediction mask using morphological operations and size filtering"""
    # Convert to uint8
    mask = (pred_mask * 255).astype(np.uint8)
    
    # Apply morphological operations to close small holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create a new mask with only components larger than min_size
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask > 0

# Function to find the optimal threshold on validation set
def find_optimal_threshold(model, val_loader, device):
    """Find the optimal threshold for converting probabilities to binary predictions"""
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_dice = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        dice_scores = []
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > threshold).float()
                
                # Calculate Dice for this batch
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                batch_dice = (2 * intersection + 1e-5) / (union + 1e-5)
                dice_scores.extend(batch_dice.cpu().numpy())
        
        avg_dice = np.mean(dice_scores)
        print(f"Threshold {threshold:.2f}, Average Dice: {avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f} with Dice: {best_dice:.4f}")
    return best_threshold

# Evaluation metrics function
def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for segmentation evaluation"""
    # Flatten the arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Calculate Dice coefficient (same as F1 for binary case)
    dice = f1
    
    # Calculate IoU (Intersection over Union)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'dice': dice,
        'iou': iou
    }

# Initialize model and training components
model = ImprovedEXOCHIP(num_classes=1).to(device)

# Use a lower learning rate for medical image segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Use ReduceLROnPlateau scheduler to reduce learning rate when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Initialize combined loss with weighting based on class imbalance
loss_function = CombinedLoss(class_ratio=class_ratio)

# Training loop with validation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    """Train the model with early stopping and validation"""
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    # To track metrics
    train_losses = []
    val_losses = []
    val_metrics = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        batch_count = 0
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                true_masks = masks.cpu().numpy() > 0.5
                
                # Calculate batch metrics
                batch_metrics = calculate_metrics(true_masks, preds)
                all_metrics.append(batch_metrics)
                
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        val_losses.append(avg_val_loss)
        
        # Average metrics across batches
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        val_metrics.append(avg_metrics)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Metrics: Dice={avg_metrics['dice']:.4f}, IoU={avg_metrics['iou']:.4f}, Accuracy={avg_metrics['accuracy']:.4f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        print("-" * 50)
    
    # Load best model
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, train_losses, val_losses, val_metrics

# Run training with 50 epochs and early stopping with patience of 10
num_epochs = 50
trained_model, train_losses, val_losses, val_metrics = train_model(
    model, train_dataloader, val_dataloader, loss_function, optimizer, scheduler, 
    num_epochs=num_epochs, patience=10
)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([m['dice'] for m in val_metrics], label='Dice')
plt.plot([m['iou'] for m in val_metrics], label='IoU')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Metrics')
plt.legend()
plt.tight_layout()
plt.show()

# Find optimal threshold using validation set
threshold = find_optimal_threshold(model, val_dataloader, device)

# Final evaluation on test set
def evaluate_model(model, test_loader, threshold=0.5, post_process=True):
    """Evaluate model on test set"""
    model.eval()
    all_metrics = []
    sample_images = []
    sample_masks = []
    sample_preds = []
    sample_preds_processed = []
    samples_collected = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            # Take the first few samples for visualization
            if samples_collected < 5:
                sample_images.extend(images.cpu())
                sample_masks.extend(masks.cpu())
                samples_collected += images.shape[0]
            
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy()
            true_masks = masks.cpu().numpy() > 0.5
            
            # Apply post-processing if enabled
            if post_process:
                processed_preds = np.array([post_process_mask(p[0]) for p in preds])
                
                # Save processed predictions for visualization
                if len(sample_preds) < 5:
                    sample_preds.extend([p[0] for p in preds])
                    sample_preds_processed.extend(processed_preds)
                
                # Calculate metrics on processed predictions
                batch_metrics = calculate_metrics(true_masks, processed_preds[:, np.newaxis, :, :])
            else:
                # Save raw predictions for visualization
                if len(sample_preds) < 5:
                    sample_preds.extend([p[0] for p in preds])
                    sample_preds_processed.extend([p[0] for p in preds])  # Same as original for display
                
                # Calculate metrics on raw predictions
                batch_metrics = calculate_metrics(true_masks, preds)
            
            all_metrics.append(batch_metrics)
    
    # Average metrics across all test batches
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print("\nTest Set Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    return avg_metrics, sample_images, sample_masks, sample_preds, sample_preds_processed

# Evaluate with and without post-processing
metrics_raw, _, _, _, _ = evaluate_model(model, test_dataloader, threshold=threshold, post_process=False)
metrics_processed, sample_images, sample_masks, sample_preds, sample_preds_processed = evaluate_model(
    model, test_dataloader, threshold=threshold, post_process=True
)

# Compare metrics with and without post-processing
print("\nMetrics Comparison (Raw vs Post-processed):")
for metric in metrics_raw.keys():
    print(f"{metric.capitalize()}: {metrics_raw[metric]:.4f} vs {metrics_processed[metric]:.4f}")

# Visualize results
def visualize_results(images, masks, preds, processed_preds, num_samples=3):
    """Visualize sample predictions"""
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        plt.figure(figsize=(16, 4))
        
        # Original image
        plt.subplot(1, 4, 1)
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")
        
        # Ground truth
        plt.subplot(1, 4, 2)
        plt.imshow(masks[i][0].numpy(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
        
        # Raw prediction
        plt.subplot(1, 4, 3)
        plt.imshow(preds[i], cmap="gray")
        plt.title(f"Raw Prediction (t={threshold:.2f})")
        plt.axis("off")
        
        # Post-processed prediction
        plt.subplot(1, 4, 4)
        plt.imshow(processed_preds[i], cmap="gray")
        plt.title("Post-processed Prediction")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()

# Visualize sample results
visualize_results(sample_images, sample_masks, sample_preds, sample_preds_processed, num_samples=3)

# # Confusion matrix visualization
# def plot_confusion_matrix(y_true, y_pred):
