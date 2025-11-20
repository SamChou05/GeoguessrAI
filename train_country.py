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
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001]
WEIGHT_DECAY = 0.0001

# Paths
TRAIN_OUT = 'train_out'
MODEL = 'model'
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
LOCAL_TRAIN_DIR = 'kaggle_train'  # Kaggle training dataset (split from full dataset)
LOCAL_TEST_DIR = 'kaggle_test'    # Temporary test split (replace with your separate test dataset later)

# S3 client (only initialized if using S3)
s3_client = None
if USE_S3:
    if boto3 is None:
        raise ImportError("boto3 is required for S3 support. Install with: pip install boto3")
    s3_client = boto3.client('s3')

start_time = perf_counter()

def preprocess(image):
    """Preprocess image for ResNet-18"""
    if image.shape[0] == 4:  # remove alpha channel
        image = image[:3]
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
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
    
    def __init__(self, image_paths, countries, country_to_idx):
        """
        Args:
            image_paths: List of image file paths (S3 keys or local paths)
            countries: List of country names corresponding to each image
            country_to_idx: Dictionary mapping country name to class index
        """
        self.image_paths = image_paths
        self.countries = countries
        self.country_to_idx = country_to_idx
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        country = self.countries[i]
        
        # Load and preprocess image
        image = load_image_from_path(image_path)
        image = preprocess(image)
        
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
        resnet = torch.load(MODEL_RESNET)
        # Verify model matches number of classes
        if resnet.fc.out_features != num_classes:
            print(f"Warning: Model has {resnet.fc.out_features} classes, but data has {num_classes}")
            print("Reinitializing final layer...")
            num_features = resnet.fc.in_features
            resnet.fc = nn.Linear(num_features, num_classes)
    else:
        print("\nInitializing new ResNet-18 model...")
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze all layers except final layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Reinitialize final layer for country classification
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        print(f"Model initialized with {num_classes} output classes")
    
    # Create datasets and data loaders
    train_data = CountryDataset(train_image_paths, train_countries, country_to_idx)
    test_data = CountryDataset(test_image_paths, test_countries, country_to_idx)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    
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
    if os.path.exists(LOSSES_CSV):
        with open(LOSSES_CSV, 'r') as losses_csv:
            next(losses_csv)
            for row in losses_csv:
                epoch += 1
                train_loss, test_loss = eval(row)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
    
    train_accuracies = []
    test_accuracies = []
    if os.path.exists(ACCURACIES_CSV):
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
    
    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {MODEL_RESNET}")
    print(f"Country mapping saved to: {MODEL_COUNTRY_MAP}")
    print("=" * 60)

if __name__ == '__main__':
    main()

