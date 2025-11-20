#!/usr/bin/env python3
"""
Split images from organized folders into train and test sets.

Takes a directory with country folders and splits each country's images
into train/test folders.
"""

import os
import shutil
import random
import argparse
from collections import defaultdict

def split_train_test(source_dir, train_dir, test_dir, test_ratio=0.1, seed=42):
    """
    Split images from source into train/test folders.
    
    Args:
        source_dir: Source directory with country folders
        train_dir: Target directory for training data
        test_dir: Target directory for test data
        test_ratio: Fraction of images to use for testing (default 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print(f"Splitting data from {source_dir}")
    print(f"Train: {train_dir}")
    print(f"Test: {test_dir}")
    print(f"Test ratio: {test_ratio}")
    print()
    
    stats = defaultdict(lambda: {'train': 0, 'test': 0})
    
    for country_folder in sorted(os.listdir(source_dir)):
        country_path = os.path.join(source_dir, country_folder)
        if not os.path.isdir(country_path):
            continue
        
        # Get all images
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]
        
        if len(images) == 0:
            print(f"  {country_folder}: No images found, skipping")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - test_ratio))
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Ensure at least 1 image in test set if we have enough
        if len(images) > 1 and len(test_images) == 0:
            test_images = [train_images.pop()]
        
        # Create directories
        train_country_dir = os.path.join(train_dir, country_folder)
        test_country_dir = os.path.join(test_dir, country_folder)
        os.makedirs(train_country_dir, exist_ok=True)
        os.makedirs(test_country_dir, exist_ok=True)
        
        # Copy images
        for img in train_images:
            shutil.copy2(
                os.path.join(country_path, img),
                os.path.join(train_country_dir, img)
            )
            stats[country_folder]['train'] += 1
        
        for img in test_images:
            shutil.copy2(
                os.path.join(country_path, img),
                os.path.join(test_country_dir, img)
            )
            stats[country_folder]['test'] += 1
        
        print(f"  {country_folder}: {len(train_images)} train, {len(test_images)} test")
    
    print()
    print("=" * 60)
    print("Split Complete")
    print("=" * 60)
    total_train = sum(s['train'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    print(f"Total: {total_train} train, {total_test} test images")
    print(f"Countries: {len(stats)}")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split images into train/test sets')
    parser.add_argument('source_dir', help='Source directory with country folders')
    parser.add_argument('train_dir', help='Target directory for training data')
    parser.add_argument('test_dir', help='Target directory for test data')
    parser.add_argument('--test-ratio', type=float, default=0.1, 
                       help='Fraction of images for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_train_test(args.source_dir, args.train_dir, args.test_dir, 
                    args.test_ratio, args.seed)

