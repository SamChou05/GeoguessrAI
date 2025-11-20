# Training Your GeoGuessr AI Model

This guide covers training a GeoGuessr AI model that predicts which country a Street View image is from. Instead of guessing exact coordinates, we're teaching the model to recognize countries - like distinguishing France from Japan. More intuitive and easier to understand.

## Setting Up Your Data

Organize your images in folders, one folder per country. Something like:

data/train/United_States/image1.jpg
data/train/United_States/image2.jpg
data/train/Canada/image1.jpg
data/test/United_States/image1.jpg
data/test/Canada/image1.jpg

So you have a train folder and a test folder, and inside each you have country folders with images inside them. The folder name becomes the label automatically. If you're using S3, same idea - just organize it the same way in your bucket.

## Installation

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn boto3
```

Update paths in `train_country.py`:

```python
USE_S3 = False
LOCAL_TRAIN_DIR = 'data/train'
LOCAL_TEST_DIR = 'data/test'
```

For S3, configure AWS:

```bash
aws configure
```

Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## Training

Run:

```bash
python3 train_country.py
```

It discovers countries from folder names, creates the mapping, loads images, and starts training. Progress updates show loss and accuracy per batch. Model saves to `model/resnet_country.pt`, country mapping to `model/country_map.pkl`.

To adjust training, edit these in `train_country.py`:

```python
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001]
WEIGHT_DECAY = 0.0001
```

If you run out of memory, reduce BATCH_SIZE to 16 or 8.

## Testing Predictions

Put test images in `demo_in/` and run:

```bash
python3 demo_country.py
```

Or use it in code:

```python
from demo_country import GeoguessrAICountry

ai = GeoguessrAICountry()
country, confidence = ai.guess('path/to/image.jpg')
print(f"Predicted: {country} ({confidence:.2%})")

# Get top 3 guesses
top_3 = ai.guess_with_top_k('path/to/image.jpg', k=3)
for country, conf in top_3:
    print(f"{country}: {conf:.2%}")
```

## Evaluation

To evaluate performance:

```bash
python3 test_country.py
```

This tests on your test set, shows overall and per-country accuracy, and generates visualizations saved to `test_out/`. The confusion matrix shows which countries get confused - like Canada and United States, which makes sense.

## Differences from Original Approach

The original `train.py` predicts exact coordinates using 21 geographic regions. This version predicts country names directly. Number of classes equals the number of countries in your dataset. Results are more interpretable - you get "France" instead of wondering why it guessed a specific coordinate.

## Tips

Use consistent country naming. Pick "United_States" and stick with it - don't mix "USA", "US", "United States". The model learns from folder names.

Try to balance your data. Having 10,000 images of the US but only 50 of Bhutan will bias the model. Aim for roughly similar amounts per country.

More data helps. At least 100-200 images per country is recommended, but more is better.

Supported formats: .png, .jpg, .jpeg

Training can be slow at first while it loads images, but once batches start processing you'll see steady progress.

The script saves progress, so you can stop and resume training if needed.

## Troubleshooting

**S3 access issues:** Check AWS credentials with `aws sts get-caller-identity`, verify bucket permissions, make sure bucket name is correct.

**Out of memory:** Reduce BATCH_SIZE in `train_country.py` or close other programs.

**Low accuracy:** Check that images are labeled correctly, ensure enough samples per country (100+ recommended), try training for more epochs, or consider that some countries are harder to distinguish.

**Country not found errors:** Make sure country names match exactly between train and test sets. Check for typos or extra spaces. "United_States" ≠ "USA" ≠ "United States".

## Example Workflow

```bash
# Organize your data
mkdir -p data/train data/test
# Copy images to country folders...

# Update paths in train_country.py
# Then train
python3 train_country.py

# Test it
python3 demo_country.py

# Evaluate
python3 test_country.py
```

Most issues are path or naming problems - once those are sorted out, you should be good to go.
