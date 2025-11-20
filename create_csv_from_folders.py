#!/usr/bin/env python3
"""
Create a CSV file from folder structure.

Given a folder structure like:
data/train/
  United_States/
    image1.jpg
    image2.jpg
  Canada/
    image3.jpg

Creates CSV:
image_path,country
train/United_States/image1.jpg,United_States
train/United_States/image2.jpg,United_States
train/Canada/image3.jpg,Canada
"""

import pandas as pd
import os
import argparse
from pathlib import Path

def create_csv_from_folders(base_dir, output_csv, relative_to=None):
    """
    Create CSV from folder structure.
    
    Args:
        base_dir: Base directory containing country folders
        output_csv: Path to output CSV file
        relative_to: If provided, make paths relative to this directory
    """
    data = []
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    print(f"Scanning {base_dir}...")
    
    for country_folder in sorted(os.listdir(base_dir)):
        country_path = os.path.join(base_dir, country_folder)
        if not os.path.isdir(country_path):
            continue
        
        images = [f for f in os.listdir(country_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]
        
        for filename in sorted(images):
            image_path = os.path.join(country_path, filename)
            
            # Make path relative if requested
            if relative_to:
                image_path = os.path.relpath(image_path, relative_to)
            
            data.append({
                'image_path': image_path,
                'country': country_folder
            })
        
        print(f"  {country_folder}: {len(images)} images")
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print()
    print("=" * 60)
    print(f"Created {output_csv}")
    print(f"Total images: {len(df)}")
    print(f"Countries: {len(df['country'].unique())}")
    print("=" * 60)
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CSV from folder structure')
    parser.add_argument('base_dir', help='Base directory containing country folders')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument('--relative-to', help='Make paths relative to this directory')
    
    args = parser.parse_args()
    
    create_csv_from_folders(args.base_dir, args.output_csv, args.relative_to)

