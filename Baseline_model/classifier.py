import os
import argparse

import torch
from torchvision import transforms
import torch.nn as nn

from PIL import Image

from main import Autoencoder

# Function to load the model
def load_model(model_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((496, 496)),  # Resize to the input size used during training
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to classify the image
def classify_image(model, image_tensor):
    with torch.no_grad():
        _, classified = model(image_tensor)
    return classified

# Function to get the prediction
def get_prediction(classified_output, threshold=0.2):
    probabilities = torch.sigmoid(classified_output)
    prediction = (probabilities > threshold).float()
    return prediction.item()

# Main function to classify images in a directory
def main():
    parser = argparse.ArgumentParser(description='Classify images using the trained model.')
    parser.add_argument('--version', type=int, required=True, help='Version of the model to use. (integer value of the version)')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold for classification')
    args = parser.parse_args()
    
    model_path = f"C:\Users\runed\Documents\Dokumenter\Uni\Master_thesis\Code\Code\code_outputs\models\simple_ae_{args.version}"  # Update with the path to your model
    image_dir = r"C:\Users\runed\Documents\Dokumenter\Uni\Master_thesis\Spectrogram\Error"  # Update with the path to your image directory
    
    model = load_model(model_path)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_tensor = preprocess_image(image_path)
        
        classified_output = classify_image(model, image_tensor)
        
        prediction = get_prediction(classified_output, threshold=args.threshold)
        
        print(f'Image: {image_file}, Prediction: {prediction}')

if __name__ == "__main__":
    main()

