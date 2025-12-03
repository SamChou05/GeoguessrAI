"""
Demo script for country-based GeoGuessr AI.
Predicts the country of a given Street View image.
"""

from train_country import preprocess, MODEL_COUNTRY_MAP, MODEL_RESNET, load_image_from_path
import torch
from torchvision import io
import os
import pickle

DEMO_IN = 'demo_in'

class GeoguessrAICountry():
    """Country-based GeoGuessr AI"""
    
    def __init__(self):
        """Load the trained model and country mapping"""
        # Load country mapping
        with open(MODEL_COUNTRY_MAP, 'rb') as f:
            mapping = pickle.load(f)
            self.country_to_idx = mapping['country_to_idx']
            self.idx_to_country = mapping['idx_to_country']
            self.num_classes = mapping['num_classes']
        
        # Load model
        self.model = torch.load(MODEL_RESNET, weights_only=False)
        self.model.eval()

        # Handle both Sequential (with dropout) and Linear fc layers
        if hasattr(self.model.fc, 'out_features'):
            model_classes = self.model.fc.out_features
        else:
            # Sequential: Dropout -> Linear, so Linear is at index 1
            model_classes = self.model.fc[1].out_features

        assert model_classes == self.num_classes, \
            f"Model has {model_classes} classes but mapping has {self.num_classes}"
        
        print(f"Loaded model with {self.num_classes} countries")
        print(f"Countries: {sorted(self.country_to_idx.keys())}")
    
    def guess(self, image_path):
        """
        Predict the country for a given image.
        
        Args:
            image_path: Path to image file (local or S3 key)
        
        Returns:
            tuple: (country_name, confidence_score)
        """
        # Load and preprocess image
        image = load_image_from_path(image_path)
        image = preprocess(image)
        
        # Make prediction
        with torch.no_grad():
            pred = self.model(image.unsqueeze(0))
            probabilities = torch.softmax(pred, dim=1)
            predicted_idx = torch.argmax(pred).item()
            confidence = probabilities[0][predicted_idx].item()
        
        country = self.idx_to_country[predicted_idx]
        return country, confidence
    
    def guess_with_top_k(self, image_path, k=5):
        """
        Predict top-k countries for a given image.
        
        Args:
            image_path: Path to image file
            k: Number of top predictions to return
        
        Returns:
            list: [(country_name, confidence_score), ...]
        """
        image = load_image_from_path(image_path)
        image = preprocess(image)
        
        with torch.no_grad():
            pred = self.model(image.unsqueeze(0))
            probabilities = torch.softmax(pred, dim=1)
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
        
        results = []
        for i in range(k):
            idx = top_k_indices[0][i].item()
            prob = top_k_probs[0][i].item()
            country = self.idx_to_country[idx]
            results.append((country, prob))
        
        return results

def main():
    """Run demo on images in demo_in folder"""
    geoguessr_ai = GeoguessrAICountry()
    
    if not os.path.exists(DEMO_IN):
        print(f"Error: {DEMO_IN} directory not found")
        return
    
    image_files = [f for f in os.listdir(DEMO_IN) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {DEMO_IN}/")
        return
    
    print(f"\nProcessing {len(image_files)} images...\n")
    print("=" * 60)
    
    for filename in sorted(image_files):
        image_path = os.path.join(DEMO_IN, filename)
        
        try:
            # Get top prediction
            country, confidence = geoguessr_ai.guess(image_path)
            
            # Get top 3 predictions
            top_3 = geoguessr_ai.guess_with_top_k(image_path, k=3)
            
            print(f"\n{filename}:")
            print(f"  Predicted Country: {country} (confidence: {confidence:.2%})")
            print(f"  Top 3 predictions:")
            for i, (c, conf) in enumerate(top_3, 1):
                print(f"    {i}. {c}: {conf:.2%}")
            print()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("=" * 60)
    print("Done!")

if __name__ == '__main__':
    main()

