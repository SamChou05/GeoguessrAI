#!/usr/bin/env python3
"""
Organize images from a CSV file into country folders.

CSV format:
image_path,country
path/to/image1.jpg,United_States
path/to/image2.jpg,Canada
"""

import pandas as pd
import shutil
import os
import argparse
from pathlib import Path

def organize_from_csv(csv_path, source_dir, target_dir, copy=True):
    """
    Organize images from CSV into country folders.
    
    Args:
        csv_path: Path to CSV file with image_path and country columns
        source_dir: Base directory where images currently are
        target_dir: Target directory for organized structure (data/train or data/test)
        copy: If True, copy files. If False, move files.
    """
    df = pd.read_csv(csv_path)
    
    # Validate CSV has required columns
    required_cols = ['image_path', 'country']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must have columns: {required_cols}")
    
    print(f"Organizing {len(df)} images from {csv_path}")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print()
    
    stats = {}
    copied = 0
    missing = 0
    
    for idx, row in df.iterrows():
        image_path = row['image_path']
        country = row['country']
        
        # Create country folder
        country_dir = os.path.join(target_dir, country)
        os.makedirs(country_dir, exist_ok=True)
        
        # Track stats
        if country not in stats:
            stats[country] = 0
        
        # Build source and target paths
        if os.path.isabs(image_path):
            source = image_path
        else:
            source = os.path.join(source_dir, image_path)
        
        filename = os.path.basename(image_path)
        target = os.path.join(country_dir, filename)
        
        # Copy or move image
        if os.path.exists(source):
            if copy:
                shutil.copy2(source, target)
            else:
                shutil.move(source, target)
            stats[country] += 1
            copied += 1
            
            if (copied + missing) % 100 == 0:
                print(f"  Processed {copied + missing}/{len(df)} images...")
        else:
            missing += 1
            print(f"  Warning: {source} not found")
    
    print()
    print("=" * 60)
    print("Organization Complete")
    print("=" * 60)
    print(f"Total images processed: {copied + missing}")
    print(f"Successfully {'copied' if copy else 'moved'}: {copied}")
    print(f"Missing files: {missing}")
    print()
    print("Images per country:")
    for country in sorted(stats.keys()):
        print(f"  {country}: {stats[country]} images")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize images from CSV into country folders')
    parser.add_argument('csv', help='Path to CSV file with image_path and country columns')
    parser.add_argument('--source', default='.', help='Base directory where images are located')
    parser.add_argument('--target', required=True, help='Target directory (e.g., data/train or data/test)')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying')
    
    args = parser.parse_args()
    
    organize_from_csv(
        args.csv,
        args.source,
        args.target,
        copy=not args.move
    )

