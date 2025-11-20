#!/usr/bin/env python3
"""
Prepare Kaggle GeoGuessr dataset for training.

This script helps download and organize the Kaggle dataset:
https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k

The dataset should be organized as:
dataset/
├── Country1/
│   ├── image1.jpg
│   └── ...
├── Country2/
│   └── ...
└── ...

This script will:
1. Validate the dataset structure
2. Optionally split into train/test
3. Verify everything is ready for training
"""

import os
import argparse
import shutil
from pathlib import Path
from validate_dataset import validate_dataset

def check_kaggle_structure(dataset_dir):
    """
    Check if the dataset is in the expected Kaggle format.
    Expected: dataset_dir/Country/image.jpg
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return False
    
    countries = []
    total_images = 0
    
    print(f"Checking dataset structure in {dataset_dir}...")
    
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            # Count images in this country folder
            images = [f for f in os.listdir(item_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]
            if images:
                countries.append(item)
                total_images += len(images)
                print(f"  ✓ {item}: {len(images)} images")
    
    print(f"\nFound {len(countries)} countries with {total_images} total images")
    return len(countries) > 0

def prepare_for_training(dataset_dir, output_dir='data', test_ratio=0.1, seed=42):
    """
    Prepare Kaggle dataset for training by splitting into train/test.
    
    Args:
        dataset_dir: Directory containing country folders with images
        output_dir: Output directory (will create train/ and test/ subdirectories)
        test_ratio: Fraction of images to use for testing
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    print("=" * 60)
    print("Preparing Kaggle Dataset for Training")
    print("=" * 60)
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Test ratio: {test_ratio}")
    print()
    
    # Check structure
    if not check_kaggle_structure(dataset_dir):
        print("Error: Dataset structure is invalid")
        return False
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each country
    stats = {}
    for country_folder in sorted(os.listdir(dataset_dir)):
        country_path = os.path.join(dataset_dir, country_folder)
        if not os.path.isdir(country_path):
            continue
        
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]
        
        if len(images) == 0:
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - test_ratio))
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Ensure at least 1 test image if we have enough
        if len(images) > 1 and len(test_images) == 0:
            test_images = [train_images.pop()]
        
        # Create country directories
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
        
        for img in test_images:
            shutil.copy2(
                os.path.join(country_path, img),
                os.path.join(test_country_dir, img)
            )
        
        stats[country_folder] = {
            'train': len(train_images),
            'test': len(test_images),
            'total': len(images)
        }
        
        print(f"  {country_folder}: {len(train_images)} train, {len(test_images)} test")
    
    print()
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    total_train = sum(s['train'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    print(f"Total: {total_train} train, {total_test} test images")
    print(f"Countries: {len(stats)}")
    print()
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print()
    print("Next steps:")
    print("  1. Review the dataset: python3 validate_dataset.py data/train data/test")
    print("  2. Configure train_country.py (set LOCAL_TRAIN_DIR and LOCAL_TEST_DIR)")
    print("  3. Train: python3 train_country.py")
    print("=" * 60)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Prepare Kaggle GeoGuessr dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Check dataset structure
  python3 prepare_kaggle_dataset.py --check /path/to/kaggle/dataset
  
  # Prepare dataset (split into train/test)
  python3 prepare_kaggle_dataset.py /path/to/kaggle/dataset --output data
  
  # Custom test ratio
  python3 prepare_kaggle_dataset.py /path/to/kaggle/dataset --output data --test-ratio 0.2
        """
    )
    parser.add_argument('dataset_dir', nargs='?', 
                       help='Directory containing Kaggle dataset (country folders)')
    parser.add_argument('--check', action='store_true',
                       help='Only check dataset structure, do not prepare')
    parser.add_argument('--output', default='data',
                       help='Output directory for train/test split (default: data)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Fraction of images for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.check:
        if not args.dataset_dir:
            parser.error("--check requires dataset_dir")
        check_kaggle_structure(args.dataset_dir)
    else:
        if not args.dataset_dir:
            parser.error("dataset_dir is required")
        prepare_for_training(args.dataset_dir, args.output, args.test_ratio, args.seed)

if __name__ == '__main__':
    main()

