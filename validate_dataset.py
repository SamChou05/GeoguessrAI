#!/usr/bin/env python3
"""
Validate dataset structure and print statistics.
"""

import os
import argparse
from collections import defaultdict

def validate_dataset(train_dir, test_dir=None):
    """Validate dataset structure and print statistics"""
    
    def count_images(base_dir):
        stats = defaultdict(int)
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} does not exist")
            return stats
        
        for country_folder in sorted(os.listdir(base_dir)):
            country_path = os.path.join(base_dir, country_folder)
            if os.path.isdir(country_path):
                count = len([f for f in os.listdir(country_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))])
                if count > 0:
                    stats[country_folder] = count
        return stats
    
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    
    train_stats = count_images(train_dir)
    print(f"\nTraining Data ({train_dir}):")
    total_train = 0
    for country, count in sorted(train_stats.items()):
        print(f"  {country}: {count} images")
        total_train += count
    print(f"  Total: {total_train} images across {len(train_stats)} countries")
    
    if test_dir:
        test_stats = count_images(test_dir)
        print(f"\nTest Data ({test_dir}):")
        total_test = 0
        for country, count in sorted(test_stats.items()):
            print(f"  {country}: {count} images")
            total_test += count
        print(f"  Total: {total_test} images across {len(test_stats)} countries")
        
        # Check for overlap
        train_countries = set(train_stats.keys())
        test_countries = set(test_stats.keys())
        missing_in_test = train_countries - test_countries
        missing_in_train = test_countries - train_countries
        
        if missing_in_test:
            print(f"\n⚠ Warning: Countries in train but not in test: {sorted(missing_in_test)}")
        if missing_in_train:
            print(f"⚠ Warning: Countries in test but not in train: {sorted(missing_in_train)}")
        
        # Check for balanced splits
        print(f"\nTrain/Test Split Ratios:")
        common_countries = train_countries & test_countries
        for country in sorted(common_countries):
            train_count = train_stats[country]
            test_count = test_stats[country]
            total = train_count + test_count
            ratio = test_count / total if total > 0 else 0
            print(f"  {country}: {train_count} train, {test_count} test ({ratio:.1%} test)")
    
    # Check for minimum samples
    print(f"\nMinimum Samples Check:")
    min_samples = 100  # Recommended minimum
    low_countries = []
    all_stats = train_stats.copy()
    if test_dir:
        for country, count in test_stats.items():
            all_stats[country] = all_stats.get(country, 0) + count
    
    for country, count in sorted(all_stats.items()):
        if count < min_samples:
            low_countries.append((country, count))
            print(f"  ⚠ {country}: {count} images (recommended: {min_samples}+)")
    
    if not low_countries:
        print(f"  ✅ All countries have at least {min_samples} images")
    
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate dataset structure')
    parser.add_argument('train_dir', help='Training data directory')
    parser.add_argument('test_dir', nargs='?', help='Test data directory (optional)')
    
    args = parser.parse_args()
    
    validate_dataset(args.train_dir, args.test_dir)

