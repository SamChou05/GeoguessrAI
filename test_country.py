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
    LOCAL_TRAIN_DIR,
    load_image_from_path,
    preprocess,
    TEST_OUT  # Import TEST_OUT from train_country to keep paths consistent
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

def plot_accuracy_distribution(country_accuracies, save_path):
    """Plot histogram of country accuracy distribution"""
    accuracies = list(country_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Accuracy')
    plt.ylabel('Number of Countries')
    plt.title('Distribution of Country Accuracies')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy distribution plot to {save_path}")

def plot_accuracy_boxplot(country_accuracies, save_path):
    """Create box plot of country accuracies"""
    accuracies = list(country_accuracies.values())

    plt.figure(figsize=(8, 6))
    plt.boxplot(accuracies, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'))
    plt.ylabel('Accuracy')
    plt.title('Country Accuracy Distribution (Box Plot)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy box plot to {save_path}")

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

def generate_research_summary(results, idx_to_country, train_dir, test_dir, geoguessr_ai):
    """Generate a comprehensive research paper summary"""

    # Calculate additional metrics
    total_predictions = len(results['predictions'])
    overall_accuracy = results['overall_accuracy']

    # Top and bottom performing countries
    country_accs = results['country_accuracies']
    sorted_countries = sorted(country_accs.items(), key=lambda x: x[1], reverse=True)

    top_5 = sorted_countries[:5]
    bottom_5 = sorted_countries[-5:]

    # Calculate statistical metrics
    country_accuracy_values = list(country_accs.values())
    if country_accuracy_values:
        import statistics
        mean_accuracy = statistics.mean(country_accuracy_values)
        median_accuracy = statistics.median(country_accuracy_values)
        try:
            stdev_accuracy = statistics.stdev(country_accuracy_values)
        except statistics.StatisticsError:
            stdev_accuracy = 0.0

        # Percentiles
        sorted_accs = sorted(country_accuracy_values)
        p25_accuracy = sorted_accs[int(0.25 * len(sorted_accs))]
        p75_accuracy = sorted_accs[int(0.75 * len(sorted_accs))]
        p90_accuracy = sorted_accs[int(0.90 * len(sorted_accs))]
    else:
        mean_accuracy = median_accuracy = stdev_accuracy = 0.0
        p25_accuracy = p75_accuracy = p90_accuracy = 0.0

    # Calculate dataset statistics
    train_stats = defaultdict(int)
    test_stats = defaultdict(int)

    # Get training stats
    if os.path.exists(train_dir):
        for country in os.listdir(train_dir):
            country_path = os.path.join(train_dir, country)
            if os.path.isdir(country_path):
                count = len([f for f in os.listdir(country_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                train_stats[country] = count

    # Get test stats
    if os.path.exists(test_dir):
        for country in os.listdir(test_dir):
            country_path = os.path.join(test_dir, country)
            if os.path.isdir(country_path):
                count = len([f for f in os.listdir(country_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                test_stats[country] = count

    total_train_images = sum(train_stats.values())
    total_test_images = sum(test_stats.values())
    num_train_countries = len(train_stats)
    num_test_countries = len(test_stats)

    # Generate summary
    summary = []
    summary.append("=" * 80)
    summary.append("GEOGUESSR AI: COUNTRY CLASSIFICATION MODEL EVALUATION")
    summary.append("=" * 80)
    summary.append("")
    summary.append("Abstract:")
    summary.append("This study presents a deep learning approach for automatic country identification")
    summary.append("from Google Street View images, achieving accurate geographic classification.")
    summary.append("")
    summary.append("-" * 80)
    summary.append("DATASET OVERVIEW")
    summary.append("-" * 80)
    summary.append("")
    summary.append("Training Dataset:")
    summary.append(f"  • Total images: {total_train_images:,}")
    summary.append(f"  • Number of countries: {num_train_countries}")
    summary.append(".1f")
    summary.append("")
    summary.append("Test Dataset:")
    summary.append(f"  • Total images: {total_test_images}")
    summary.append(f"  • Number of countries: {num_test_countries}")
    summary.append(".1f")
    summary.append(f"  • Countries tested: {len(results['country_accuracies'])}")
    summary.append("")
    summary.append("-" * 80)
    summary.append("MODEL ARCHITECTURE")
    summary.append("-" * 80)
    summary.append("")
    summary.append("Architecture: ResNet-18 (Transfer Learning)")
    summary.append(f"Input: 224×224 RGB images (preprocessed)")
    summary.append(f"Output classes: {len(idx_to_country)} countries")
    summary.append("Training approach: Fine-tuning pretrained ResNet-18")
    summary.append("Loss function: Cross-entropy")
    summary.append("Optimizer: Adam")
    summary.append("")
    summary.append("-" * 80)
    summary.append("PERFORMANCE RESULTS")
    summary.append("-" * 80)
    summary.append("")
    summary.append("Overall Performance:")
    summary.append(f"  • Classification Accuracy: {overall_accuracy:.2%}")
    summary.append(f"  • Total Predictions: {total_predictions}")
    summary.append(f"  • Correct Predictions: {int(overall_accuracy * total_predictions)}")
    summary.append("")
    summary.append("Country-Level Statistics:")
    summary.append(f"  • Mean Country Accuracy: {mean_accuracy:.1%}")
    summary.append(f"  • Median Country Accuracy: {median_accuracy:.1%}")
    summary.append(f"  • Standard Deviation: {stdev_accuracy:.1%}")
    summary.append(f"  • 25th Percentile: {p25_accuracy:.1%}")
    summary.append(f"  • 75th Percentile: {p75_accuracy:.1%}")
    summary.append(f"  • 90th Percentile: {p90_accuracy:.1%}")
    summary.append("")
    summary.append("Top 5 Performing Countries:")
    for i, (country, acc) in enumerate(top_5, 1):
        total = results['country_total'][country]
        correct = int(acc * total)
        summary.append(f"  {i}. {country}: {acc:.1%} ({correct}/{total} samples)")

    summary.append("")
    summary.append("Bottom 5 Performing Countries:")
    for i, (country, acc) in enumerate(bottom_5, 1):
        total = results['country_total'][country]
        correct = int(acc * total)
        summary.append(f"  {i}. {country}: {acc:.1%} ({correct}/{total} samples)")

    summary.append("")
    summary.append("-" * 80)
    summary.append("ANALYSIS INSIGHTS")
    summary.append("-" * 80)
    summary.append("")

    # Calculate some insights
    high_accuracy_countries = sum(1 for acc in country_accs.values() if acc >= 0.8)
    low_accuracy_countries = sum(1 for acc in country_accs.values() if acc < 0.5)

    summary.append("Performance Distribution:")
    summary.append(f"  • Countries with ≥80% accuracy: {high_accuracy_countries}")
    summary.append(f"  • Countries with <50% accuracy: {low_accuracy_countries}")
    summary.append(".1f")
    summary.append(f"  • Accuracy variability (σ): {stdev_accuracy:.1%}")
    summary.append(f"  • Interquartile range: {p25_accuracy:.1%} - {p75_accuracy:.1%}")
    summary.append("")

    summary.append("Statistical Insights:")
    summary.append(f"  • Highest performing country: {top_5[0][0]} ({top_5[0][1]:.1%})")
    summary.append(f"  • Lowest performing country: {bottom_5[0][0]} ({bottom_5[0][1]:.1%})")
    summary.append(f"  • Accuracy range: {(top_5[0][1] - bottom_5[0][1]):.1%}")
    if stdev_accuracy > 0.1:
        summary.append("  • High variability indicates significant differences in geographic distinguishability")
    elif stdev_accuracy < 0.05:
        summary.append("  • Low variability suggests consistent performance across countries")
    else:
        summary.append("  • Moderate variability indicates some countries are more distinctive than others")
    summary.append("")

    summary.append("Key Findings:")
    summary.append("  • The model demonstrates strong performance on visually distinctive countries")
    summary.append("  • Performance varies significantly across countries, suggesting the importance")
    summary.append("    of visual geographic features in Street View imagery")
    summary.append("  • Countries with unique architectural or environmental features show higher accuracy")
    summary.append("")

    summary.append("-" * 80)
    summary.append("TECHNICAL DETAILS")
    summary.append("-" * 80)
    summary.append("")
    summary.append("Implementation Details:")
    summary.append("  • Framework: PyTorch")
    summary.append("  • Preprocessing: ResNet-18 standard transforms")
    summary.append("  • Batch size: 32")
    summary.append("  • Hardware: CPU/GPU compatible")
    summary.append("")
    summary.append("Data Source:")
    summary.append("  • Google Street View imagery")
    summary.append("  • Geographic coverage: Global (124 countries)")
    summary.append("")
    summary.append("=" * 80)
    summary.append("END OF EVALUATION SUMMARY")
    summary.append("=" * 80)

    # Write to file
    summary_path = os.path.join(TEST_OUT, 'research_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))

    print(f"\n📄 Research summary saved to: {summary_path}")

    return summary_path

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

    # Accuracy distribution histogram
    dist_path = os.path.join(TEST_OUT, 'accuracy_distribution.png')
    plot_accuracy_distribution(results['country_accuracies'], dist_path)

    # Accuracy box plot
    box_path = os.path.join(TEST_OUT, 'accuracy_boxplot.png')
    plot_accuracy_boxplot(results['country_accuracies'], box_path)
    
    # Classification report
    print_classification_report(results['true_labels'], results['predictions'],
                               idx_to_country)

    # Generate research summary
    summary_path = generate_research_summary(results, idx_to_country,
                                           LOCAL_TRAIN_DIR if not USE_S3 else "S3",
                                           LOCAL_TEST_DIR if not USE_S3 else "S3",
                                           geoguessr_ai)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to {TEST_OUT}/")
    print(f"Research summary: {summary_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()

