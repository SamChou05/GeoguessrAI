"""
Training script for country-based classification.
Supports both S3 buckets and local folders.

Data structure expected:
- S3: s3://bucket-name/train/COUNTRY/image.jpg
- Local: data/train/COUNTRY/image.jpg

Or with CSV metadata:
- CSV with columns: image_path, country
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os
import csv
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = None

# Configuration
NUM_EPOCHS = 20  # Increased epochs with better regularization
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001, 0.00005]  # Added lower learning rate
WEIGHT_DECAY = 0.001  # Increased weight decay for regularization

# Paths - Version 2: Filtered dataset with matching countries
RUN_VERSION = 'v2_filtered'
TRAIN_OUT = f'results/run_{RUN_VERSION}/train_out'
TEST_OUT = f'results/run_{RUN_VERSION}/test_out'
MODEL = f'results/run_{RUN_VERSION}/model'
MODEL_COUNTRY_MAP = os.path.join(MODEL, 'country_map.pkl')
MODEL_RESNET = os.path.join(MODEL, 'resnet_country.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
ACCURACIES_CSV = os.path.join(TRAIN_OUT, 'accuracies.csv')
LOSS_PLOT = os.path.join(TRAIN_OUT, 'loss.png')
ACCURACY_PLOT = os.path.join(TRAIN_OUT, 'accuracy.png')

# Data source configuration
USE_S3 = False  # Set to True to use S3
S3_BUCKET = 'your-bucket-name'
S3_TRAIN_PREFIX = 'train/'
S3_TEST_PREFIX = 'test/'
LOCAL_TRAIN_DIR = 'kaggle_dataset_filtered'  # Filtered dataset (only countries with test data)
LOCAL_TEST_DIR = 'test_dataset'    # Curated test set

# S3 client (only initialized if using S3)
s3_client = None
if USE_S3:
    if boto3 is None:
        raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
    s3_client = boto3.client('s3')

start_time = perf_counter()

def generate_training_summary(total_train_images, total_test_images, num_classes, start_time):
    """Generate a comprehensive training summary for research papers"""

    training_time = perf_counter() - start_time

    # Load training history for detailed metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    if os.path.exists(LOSSES_CSV) and os.path.getsize(LOSSES_CSV) > 0:
        with open(LOSSES_CSV, 'r') as losses_csv:
            next(losses_csv)  # Skip header
            for row in losses_csv:
                train_loss, test_loss = eval(row)
                train_losses.append(train_loss)
                test_losses.append(test_loss)

    if os.path.exists(ACCURACIES_CSV) and os.path.getsize(ACCURACIES_CSV) > 0:
        with open(ACCURACIES_CSV, 'r') as accuracies_csv:
            next(accuracies_csv)  # Skip header
            for row in accuracies_csv:
                train_accuracy, test_accuracy = eval(row)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

    # Calculate detailed metrics
    final_train_accuracy = train_accuracies[-1] if train_accuracies else 0.0
    final_test_accuracy = test_accuracies[-1] if test_accuracies else 0.0
    best_train_accuracy = max(train_accuracies) if train_accuracies else 0.0
    best_test_accuracy = max(test_accuracies) if test_accuracies else 0.0
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0.0
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else 0.0

    summary = []
    summary.append("=" * 80)
    summary.append("GEOGUESSR AI: TRAINING SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    summary.append("Training Configuration:")
    summary.append(f"  • Model: ResNet-18 (Transfer Learning)")
    summary.append(f"  • Epochs: {NUM_EPOCHS}")
    summary.append(f"  • Batch Size: {BATCH_SIZE}")
    summary.append(f"  • Learning Rates: {LEARNING_RATES}")
    summary.append(f"  • Weight Decay: {WEIGHT_DECAY}")
    summary.append(f"  • Optimizer: Adam")
    summary.append("")
    summary.append("Dataset:")
    summary.append(f"  • Training Images: {total_train_images:,}")
    summary.append(f"  • Test Images: {total_test_images:,}")
    summary.append(f"  • Total Images: {total_train_images + total_test_images:,}")
    summary.append(f"  • Number of Classes: {num_classes}")
    summary.append("")
    summary.append("Training Performance:")
    summary.append(f"  • Total Training Time: {training_time:.1f} seconds ({training_time/3600:.1f} hours)")
    summary.append(f"  • Average Time per Epoch: {training_time/NUM_EPOCHS:.1f} seconds")
    summary.append("")
    summary.append("Accuracy Metrics:")
    summary.append(f"  • Final Training Accuracy: {final_train_accuracy:.2%}")
    summary.append(f"  • Final Test Accuracy: {final_test_accuracy:.2%}")
    summary.append(f"  • Best Training Accuracy: {best_train_accuracy:.2%}")
    summary.append(f"  • Best Test Accuracy: {best_test_accuracy:.2%}")
    summary.append(f"  • Average Training Accuracy: {avg_train_accuracy:.2%}")
    summary.append(f"  • Average Test Accuracy: {avg_test_accuracy:.2%}")
    summary.append("")
    summary.append("Training History:")
    summary.append(f"  • Training Epochs Completed: {len(train_accuracies)}")
    summary.append(f"  • Loss Convergence: {train_losses[-1]:.4f} (train) → {test_losses[-1]:.4f} (test)")
    summary.append("")
    summary.append("Training Dynamics:")
    if train_accuracies and test_accuracies:
        summary.append(f"  • Initial Training Accuracy: {train_accuracies[0]:.1%}")
        summary.append(f"  • Initial Test Accuracy: {test_accuracies[0]:.1%}")
        summary.append(f"  • Training Improvement: {(final_train_accuracy - train_accuracies[0]):+.1%}")
        summary.append(f"  • Test Improvement: {(final_test_accuracy - test_accuracies[0]):+.1%}")
        if len(train_accuracies) > 1:
            summary.append(f"  • Best Epoch (Test): {test_accuracies.index(best_test_accuracy) + 1}/{len(test_accuracies)}")
    summary.append("")
    summary.append("Output Files:")
    summary.append(f"  • Trained Model: {MODEL_RESNET}")
    summary.append(f"  • Country Mapping: {MODEL_COUNTRY_MAP}")
    summary.append(f"  • Loss History: {LOSSES_CSV}")
    summary.append(f"  • Accuracy History: {ACCURACIES_CSV}")
    summary.append(f"  • Loss Plot: {LOSS_PLOT}")
    summary.append(f"  • Accuracy Plot: {ACCURACY_PLOT}")
    summary.append("")
    summary.append("Next Steps:")
    summary.append("  • Run evaluation: python3 test_country.py")
    summary.append("  • Run demo: python3 demo_country.py")
    summary.append("")
    summary.append("=" * 80)

    # Write to file
    summary_path = os.path.join(TRAIN_OUT, 'training_summary.txt')
    os.makedirs(TRAIN_OUT, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))

    print(f"\n📄 Training summary saved to: {summary_path}")
    return summary_path

def get_train_transforms():
    """Get data augmentation transforms for training"""
    import torchvision.transforms as T
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    base_transforms = weights.transforms()

    # Add augmentation transforms
    train_transforms = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
        T.RandomHorizontalFlip(p=0.5),               # Random flip
        T.RandomRotation(15),                        # Random rotation
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Color jitter
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms

def get_val_transforms():
    """Get transforms for validation/testing (no augmentation)"""
    from torchvision.models import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()

def preprocess(image, is_training=False):
    """Preprocess image for ResNet-18 with optional augmentation"""
    # Convert torch.Tensor to PIL Image for transforms
    from PIL import Image
    import numpy as np

    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy array (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)  # Convert to uint8
        image = Image.fromarray(image)

    if image.mode == 'RGBA':  # remove alpha channel
        image = image.convert('RGB')

    if is_training:
        transform = get_train_transforms()
    else:
        transform = get_val_transforms()

    return transform(image)

def load_image_from_path(image_path):
    """Load image from either S3 or local filesystem"""
    if USE_S3 and s3_client:
        # Download from S3 to temporary location
        temp_path = f'/tmp/{os.path.basename(image_path)}'
        try:
            s3_client.download_file(S3_BUCKET, image_path, temp_path)
            image = io.read_image(temp_path).float()
            os.remove(temp_path)  # Clean up temp file
            return image
        except ClientError as e:
            raise FileNotFoundError(f"Failed to download {image_path} from S3: {e}")
    else:
        # Load from local filesystem
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return io.read_image(image_path).float()

def discover_countries_from_folders(base_dir, use_s3=False):
    """
    Discover all countries from folder structure.
    Returns: dict mapping country name to list of image paths
    """
    country_images = {}
    
    if use_s3:
        # List all objects in S3 with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=base_dir)
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                # Extract country from path: base_dir/COUNTRY/image.jpg
                parts = key.replace(base_dir, '').split('/')
                if len(parts) >= 2:
                    country = parts[0]
                    if country not in country_images:
                        country_images[country] = []
                    country_images[country].append(key)
    else:
        # Local filesystem
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")
        
        for country_folder in os.listdir(base_dir):
            country_path = os.path.join(base_dir, country_folder)
            if os.path.isdir(country_path):
                images = []
                for filename in os.listdir(country_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images.append(os.path.join(country_path, filename))
                if images:
                    country_images[country_folder] = images
    
    return country_images

def create_country_mapping(countries):
    """
    Create mapping from country name to class index.
    Returns: (country_to_idx dict, idx_to_country dict, num_classes)
    """
    sorted_countries = sorted(countries)
    country_to_idx = {country: idx for idx, country in enumerate(sorted_countries)}
    idx_to_country = {idx: country for country, idx in country_to_idx.items()}
    return country_to_idx, idx_to_country, len(sorted_countries)

class CountryDataset(Dataset):
    """Dataset class for country-based classification"""
    
    def __init__(self, image_paths, countries, country_to_idx, is_training=False):
        """
        Args:
            image_paths: List of image file paths (S3 keys or local paths)
            countries: List of country names corresponding to each image
            country_to_idx: Dictionary mapping country name to class index
            is_training: Whether this is for training (enables augmentation)
        """
        self.image_paths = image_paths
        self.countries = countries
        self.country_to_idx = country_to_idx
        self.is_training = is_training
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        country = self.countries[i]
        
        # Load and preprocess image (with augmentation if training)
        image = load_image_from_path(image_path)
        image = preprocess(image, is_training=self.is_training)
        
        # Get country label
        label = self.country_to_idx[country]
        
        return image, torch.LongTensor([label])

def train(model, train_loader, optimizer):
    """Training function"""
    print('training...')
    model.train()
    losses = []
    accuracies = []
    for batch, (images, labels) in enumerate(train_loader):
        labels = labels.squeeze()
        optimizer.zero_grad()
        pred = model(images)
        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
        accuracies.append(accuracy.item())
        print(f'batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item():.4f}, train accuracy: {accuracy.item():.4f}, time: {round(perf_counter() - start_time, 1)}s')
    train_loss = np.mean(losses)
    train_accuracy = np.mean(accuracies)
    return train_loss, train_accuracy

def test(model, test_loader):
    """Testing function"""
    print('testing...')
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for batch, (images, labels) in enumerate(test_loader):
            labels = labels.squeeze()
            pred = model(images)
            loss = F.cross_entropy(pred, labels)
            losses.append(loss.item())
            accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
            accuracies.append(accuracy.item())
            print(f'batch: {batch + 1}/{len(test_loader)}, test loss: {loss.item():.4f}, test accuracy: {accuracy.item():.4f}, time: {round(perf_counter() - start_time, 1)}s')
    test_loss = np.mean(losses)
    test_accuracy = np.mean(accuracies)
    return test_loss, test_accuracy

def main():
    print("=" * 60)
    print("Country-Based GeoGuessr Training")
    print("=" * 60)
    print(f"Data source: {'S3' if USE_S3 else 'Local filesystem'}")
    print(f"Train directory: {S3_TRAIN_PREFIX if USE_S3 else LOCAL_TRAIN_DIR}")
    print(f"Test directory: {S3_TEST_PREFIX if USE_S3 else LOCAL_TEST_DIR}")
    print()
    
    # Discover countries and images from train folder
    print("Discovering training data...")
    train_country_images = discover_countries_from_folders(
        S3_TRAIN_PREFIX if USE_S3 else LOCAL_TRAIN_DIR,
        use_s3=USE_S3
    )
    
    print("Discovering test data...")
    test_country_images = discover_countries_from_folders(
        S3_TEST_PREFIX if USE_S3 else LOCAL_TEST_DIR,
        use_s3=USE_S3
    )
    
    # Get countries from training data (model only learns these)
    train_countries_set = set(train_country_images.keys())
    test_countries_set = set(test_country_images.keys())
    
    # Check for countries in test that aren't in training
    test_only_countries = test_countries_set - train_countries_set
    if test_only_countries:
        print(f"\n⚠ Warning: {len(test_only_countries)} countries found in test but not in training:")
        print(f"  {sorted(test_only_countries)}")
        print("  These countries will be skipped during testing.")
        print("  Consider adding them to training data or removing from test set.")
    
    # Use only training countries for mapping (model can only predict what it was trained on)
    all_countries = train_countries_set
    print(f"\nFound {len(all_countries)} countries in training: {sorted(all_countries)}")
    if test_countries_set:
        print(f"Found {len(test_countries_set)} countries in test: {sorted(test_countries_set)}")
    
    # Create country mapping based on training countries only
    country_to_idx, idx_to_country, num_classes = create_country_mapping(all_countries)
    
    # Build train/test datasets
    train_image_paths = []
    train_countries = []
    for country, images in train_country_images.items():
        train_image_paths.extend(images)
        train_countries.extend([country] * len(images))
    
    test_image_paths = []
    test_countries = []
    skipped_test_images = 0
    for country, images in test_country_images.items():
        # Only include test images from countries that were in training
        if country in all_countries:
            test_image_paths.extend(images)
            test_countries.extend([country] * len(images))
        else:
            skipped_test_images += len(images)
    
    if skipped_test_images > 0:
        print(f"\n⚠ Skipped {skipped_test_images} test images from countries not in training set")

    # Calculate totals for summary
    total_train_images = len(train_image_paths)
    total_test_images = len(test_image_paths)

    print(f"\nTraining samples: {len(train_image_paths)}")
    print(f"Test samples: {len(test_image_paths)}")
    print(f"Number of classes (countries): {num_classes}")
    
    # Save country mapping
    if not os.path.isdir(MODEL):
        os.makedirs(MODEL)
    
    with open(MODEL_COUNTRY_MAP, 'wb') as f:
        pickle.dump({
            'country_to_idx': country_to_idx,
            'idx_to_country': idx_to_country,
            'num_classes': num_classes
        }, f)
    print(f"\nSaved country mapping to {MODEL_COUNTRY_MAP}")
    
    # Initialize or load model
    if os.path.exists(MODEL_RESNET):
        print(f"\nLoading existing model from {MODEL_RESNET}")
        resnet = torch.load(MODEL_RESNET, weights_only=False)
        # Verify model matches number of classes
        if resnet.fc.out_features != num_classes:
            print(f"Warning: Model has {resnet.fc.out_features} classes, but data has {num_classes}")
            print("Reinitializing final layer with dropout...")
            num_features = resnet.fc.in_features
            resnet.fc = nn.Sequential(
                nn.Dropout(p=0.5),  # Add dropout before final layer
                nn.Linear(num_features, num_classes)
            )
    else:
        print("\nInitializing new ResNet-18 model...")
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze all layers except final layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Reinitialize final layer for country classification with dropout
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout before final layer
            nn.Linear(num_features, num_classes)
        )
        print(f"Model initialized with {num_classes} output classes")
    
    # Create datasets and data loaders
    train_data = CountryDataset(train_image_paths, train_countries, country_to_idx, is_training=True)
    test_data = CountryDataset(test_image_paths, test_countries, country_to_idx, is_training=False)

    # Calculate class weights for balanced sampling
    from collections import Counter
    country_counts = Counter(train_countries)
    class_weights = {country_to_idx[country]: 1.0 / count for country, count in country_counts.items()}
    sample_weights = [class_weights[country_to_idx[country]] for country in train_countries]

    # Create weighted sampler for balanced training
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize output directories
    if not os.path.isdir(TRAIN_OUT):
        os.makedirs(TRAIN_OUT)
        with open(LOSSES_CSV, 'w') as losses_csv:
            loss_writer = csv.writer(losses_csv)
            loss_writer.writerow(['train_loss', 'test_loss'])
        with open(ACCURACIES_CSV, 'w') as accuracies_csv:
            accuracy_writer = csv.writer(accuracies_csv)
            accuracy_writer.writerow(['train_accuracy', 'test_accuracy'])
    
    # Load previous training state
    epoch = 0
    train_losses = []
    test_losses = []
    if os.path.exists(LOSSES_CSV) and os.path.getsize(LOSSES_CSV) > 0:
        with open(LOSSES_CSV, 'r') as losses_csv:
            next(losses_csv)
            for row in losses_csv:
                epoch += 1
                train_loss, test_loss = eval(row)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
    
    train_accuracies = []
    test_accuracies = []
    if os.path.exists(ACCURACIES_CSV) and os.path.getsize(ACCURACIES_CSV) > 0:
        with open(ACCURACIES_CSV, 'r') as accuracies_csv:
            next(accuracies_csv)
            for row in accuracies_csv:
                train_accuracy, test_accuracy = eval(row)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    with open(LOSSES_CSV, 'a') as losses_csv, open(ACCURACIES_CSV, 'a') as accuracies_csv:
        loss_writer = csv.writer(losses_csv)
        accuracy_writer = csv.writer(accuracies_csv)
        
        while epoch < NUM_EPOCHS:
            epochs_per_lr = NUM_EPOCHS // len(LEARNING_RATES)
            learning_rate = LEARNING_RATES[min(epoch // epochs_per_lr, len(LEARNING_RATES) - 1)]
            
            # Optimizer only for final layer
            optimizer = optim.Adam(resnet.fc.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
            
            train_loss, train_accuracy = train(resnet, train_loader, optimizer)
            test_loss, test_accuracy = test(resnet, test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            loss_writer.writerow([train_loss, test_loss])
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            accuracy_writer.writerow([train_accuracy, test_accuracy])
            
            torch.save(resnet, MODEL_RESNET)
            
            # Plot training curves
            plt.figure()
            plt.title('Loss vs. Epoch')
            plt.plot(range(epoch + 1), train_losses, label='Train')
            plt.plot(range(epoch + 1), test_losses, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(LOSS_PLOT)
            plt.close()
            
            plt.figure()
            plt.title('Accuracy vs. Epoch')
            plt.plot(range(epoch + 1), train_accuracies, label='Train')
            plt.plot(range(epoch + 1), test_accuracies, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(ACCURACY_PLOT)
            plt.close()
            
            print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} Summary:')
            print(f'  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            print(f'  Learning Rate: {learning_rate}')
            print()
            
            epoch += 1
    
    # Generate training summary
    training_summary = generate_training_summary(
        total_train_images, total_test_images, num_classes, start_time
    )

    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {MODEL_RESNET}")
    print(f"Country mapping saved to: {MODEL_COUNTRY_MAP}")
    print(f"Training summary saved to: {training_summary}")
    print("=" * 60)

if __name__ == '__main__':
    main()

