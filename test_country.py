"""
Test script for country-based model.
Evaluates model performance and generates visualizations.
"""

from train_country import (
    discover_countries_from_folders, 
    MODEL_COUNTRY_MAP, 
    MODEL_RESNET,
    USE_S3,
    S3_TEST_PREFIX,
    LOCAL_TEST_DIR,
    load_image_from_path,
    preprocess
)
from demo_country import GeoguessrAICountry
import torch
import numpy as np
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

TEST_OUT = 'test_out'

def evaluate_model(geoguessr_ai, test_image_paths, test_countries):
    """
    Evaluate model on test set and generate metrics.
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating on {len(test_image_paths)} test images...")
    
    correct = 0
    predictions = []
    true_labels = []
    country_correct = defaultdict(int)
    country_total = defaultdict(int)
    
    for i, (image_path, true_country) in enumerate(zip(test_image_paths, test_countries)):
        try:
            predicted_country, confidence = geoguessr_ai.guess(image_path)
            predictions.append(predicted_country)
            true_labels.append(true_country)
            
            country_total[true_country] += 1
            if predicted_country == true_country:
                correct += 1
                country_correct[true_country] += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_image_paths)} images...")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    accuracy = correct / len(test_image_paths) if test_image_paths else 0
    
    # Per-country accuracy
    country_accuracies = {}
    for country in country_total:
        country_accuracies[country] = country_correct[country] / country_total[country]
    
    return {
        'overall_accuracy': accuracy,
        'country_accuracies': country_accuracies,
        'predictions': predictions,
        'true_labels': true_labels,
        'country_total': dict(country_total)
    }

def plot_confusion_matrix(true_labels, predictions, idx_to_country, save_path):
    """Generate and save confusion matrix"""
    # Convert to indices
    country_to_idx = {v: k for k, v in idx_to_country.items()}
    true_indices = [country_to_idx[label] for label in true_labels]
    pred_indices = [country_to_idx[pred] for pred in predictions]
    
    # Create confusion matrix
    cm = confusion_matrix(true_indices, pred_indices)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_country[i] for i in range(len(idx_to_country))],
                yticklabels=[idx_to_country[i] for i in range(len(idx_to_country))])
    plt.title('Confusion Matrix')
    plt.ylabel('True Country')
    plt.xlabel('Predicted Country')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_country_accuracies(country_accuracies, save_path):
    """Plot per-country accuracy"""
    countries = sorted(country_accuracies.keys())
    accuracies = [country_accuracies[c] for c in countries]
    
    plt.figure(figsize=(12, 6))
    plt.barh(countries, accuracies)
    plt.xlabel('Accuracy')
    plt.title('Per-Country Accuracy')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved country accuracies plot to {save_path}")

def print_classification_report(true_labels, predictions, idx_to_country):
    """Print detailed classification report"""
    country_to_idx = {v: k for k, v in idx_to_country.items()}
    true_indices = [country_to_idx[label] for label in true_labels]
    pred_indices = [country_to_idx[pred] for pred in predictions]
    
    target_names = [idx_to_country[i] for i in range(len(idx_to_country))]
    report = classification_report(true_indices, pred_indices, 
                                   target_names=target_names)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

def main():
    print("=" * 60)
    print("Country-Based Model Evaluation")
    print("=" * 60)
    
    # Load model
    geoguessr_ai = GeoguessrAICountry()
    
    # Load country mapping
    with open(MODEL_COUNTRY_MAP, 'rb') as f:
        mapping = pickle.load(f)
        idx_to_country = mapping['idx_to_country']
    
    # Discover test data
    print("\nLoading test data...")
    test_country_images = discover_countries_from_folders(
        S3_TEST_PREFIX if USE_S3 else LOCAL_TEST_DIR,
        use_s3=USE_S3
    )
    
    # Build test dataset
    test_image_paths = []
    test_countries = []
    for country, images in test_country_images.items():
        test_image_paths.extend(images)
        test_countries.extend([country] * len(images))
    
    print(f"Found {len(test_image_paths)} test images across {len(test_country_images)} countries")
    
    # Create output directory
    if not os.path.isdir(TEST_OUT):
        os.makedirs(TEST_OUT)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(geoguessr_ai, test_image_paths, test_countries)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"\nPer-Country Accuracy:")
    for country in sorted(results['country_accuracies'].keys()):
        acc = results['country_accuracies'][country]
        total = results['country_total'][country]
        correct = int(acc * total)
        print(f"  {country}: {acc:.2%} ({correct}/{total} samples)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(TEST_OUT, 'confusion_matrix.png')
    plot_confusion_matrix(results['true_labels'], results['predictions'], 
                         idx_to_country, cm_path)
    
    # Country accuracies
    acc_path = os.path.join(TEST_OUT, 'country_accuracies.png')
    plot_country_accuracies(results['country_accuracies'], acc_path)
    
    # Classification report
    print_classification_report(results['true_labels'], results['predictions'], 
                               idx_to_country)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to {TEST_OUT}/")
    print("=" * 60)

if __name__ == '__main__':
    main()

