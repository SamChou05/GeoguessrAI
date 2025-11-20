# Country Training Files

Simple overview of files needed for the country-based GeoGuessr AI training system.

## Core Files

**train_country.py**
- Main training script
- Discovers countries from folder structure
- Trains ResNet-18 model to classify images by country
- Saves model to `model/resnet_country.pt` and mapping to `model/country_map.pkl`
- Supports local folders and S3 buckets

**demo_country.py**
- Inference script for making predictions
- Loads trained model and makes country predictions on images
- Can predict single country or top-k countries
- Processes images from `demo_in/` folder

**test_country.py**
- Evaluation script
- Tests model on test dataset
- Calculates accuracy (overall and per-country)
- Generates confusion matrix and accuracy plots
- Saves results to `test_out/`

## Model Files

**model/country_map.pkl**
- Country mapping dictionary
- Contains `country_to_idx`, `idx_to_country`, and `num_classes`
- Created automatically during training
- Required for inference and evaluation

**model/resnet_country.pt**
- Trained ResNet-18 model weights
- Created after training completes
- Required for inference and evaluation

## Data Structure

Your data should be organized like this:

```
kaggle_train/
  United_States/
    image1.jpg
    image2.jpg
  Canada/
    image1.jpg
  ...

kaggle_test/
  United_States/
    image1.jpg
  Canada/
    image1.jpg
  ...
```

Each folder name becomes the country label. The script discovers countries automatically from folder names.

## Usage

1. **Train**: `python3 train_country.py`
2. **Demo**: `python3 demo_country.py`
3. **Test**: `python3 test_country.py`

## Configuration

Edit these variables in `train_country.py`:

- `USE_S3`: Set to `True` for S3, `False` for local files
- `LOCAL_TRAIN_DIR`: Path to training data folder
- `LOCAL_TEST_DIR`: Path to test data folder
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATES`: Training hyperparameters

## Output Files

- `model/resnet_country.pt`: Trained model
- `model/country_map.pkl`: Country mapping
- `train_out/losses.csv`: Training loss history
- `train_out/accuracies.csv`: Training accuracy history
- `train_out/loss.png`: Loss plot
- `train_out/accuracy.png`: Accuracy plot
- `test_out/confusion_matrix.png`: Confusion matrix
- `test_out/per_country_accuracy.png`: Per-country accuracy plot

