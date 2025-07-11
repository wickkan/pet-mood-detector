"""
Gradio web interface for Pet Mood Detector.
"""
import os
import sys
import argparse
from typing import Dict, Union, List

import numpy as np
import torch
from PIL import Image
import gradio as gr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PetMoodClassifier
from src.predict import predict_image, predict_frame


def load_model(model_path: str, device: str) -> PetMoodClassifier:
    """
    Load the trained model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PetMoodClassifier.load(model_path, device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def predict_mood(image, model, class_names: List[str]) -> Dict[str, float]:
    """
    Predict pet mood from an input image.
    
    Args:
        image: Input image
        model: Trained model
        class_names: List of class names
        
    Returns:
        Dictionary mapping mood labels to confidence scores
    """
    if image is None:
        return {mood: 0.0 for mood in class_names}
    
    # Save image to a temporary file
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Get device
    device = next(model.parameters()).device
    
    # Make prediction
    result = predict_image(temp_path, model, device, class_names=class_names)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return result["probabilities"]


def setup_interface(model_path: str = "models/final_model.pth") -> gr.Interface:
    """
    Set up the Gradio interface.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Gradio interface
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['angry', 'happy', 'other', 'sad']
    
    # Load model
    try:
        model = load_model(model_path, device)
        model_loaded = True
    except FileNotFoundError:
        print(f"Warning: Model not found at {model_path}. "
              f"Please train a model first or specify the correct model path.")
        model = None
        model_loaded = False
    
    # Define examples
    examples = []
    for root, _, files in os.walk("../data"):
        if any(cls in root for cls in ['angry', 'happy', 'sad', 'other']):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('.'):
                    examples.append(os.path.join(root, file))
    examples = examples[:5]  # Limit to 5 examples
    
    # Define interface
    interface = gr.Interface(
        fn=lambda img: predict_mood(img, model, class_names) if model_loaded else {},
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=len(class_names)),
        examples=examples,
        title="🐱🐶 Pet Mood Detector",
        description=(
            "Upload an image of a cat or dog, and this model will predict "
            "its mood (happy, angry, sad, or other)."
        ),
        article=(
            "This model was trained on a dataset of pet images with different "
            "emotional expressions. It uses a pre-trained CNN backbone with "
            "transfer learning to classify pet emotions."
            "\n\n"
            "Note: The model works best with clear, front-facing images of "
            "cats and dogs where the face is clearly visible."
        ),
        allow_flagging="never"
    )
    
    return interface


def setup_webcam_interface(model_path: str = "models/final_model.pth") -> gr.Interface:
    """
    Set up the Gradio webcam interface.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Gradio interface for webcam
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['angry', 'happy', 'other', 'sad']
    
    # Load model
    try:
        model = load_model(model_path, device)
        model_loaded = True
    except FileNotFoundError:
        print(f"Warning: Model not found at {model_path}. "
              f"Please train a model first or specify the correct model path.")
        model = None
        model_loaded = False
    
    # Define webcam prediction function
    def predict_webcam(image):
        if not model_loaded or image is None:
            return {mood: 0.0 for mood in class_names}
        
        # Convert to numpy array if not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Get device
        device = next(model.parameters()).device
        
        # Make prediction
        result = predict_frame(image, model, device, class_names=class_names)
        return result["probabilities"]
    
    # Define interface
    webcam_interface = gr.Interface(
        fn=predict_webcam,
        inputs=gr.Image(source="webcam", streaming=True),
        outputs=gr.Label(num_top_classes=len(class_names)),
        live=True,
        title="🐱🐶 Live Pet Mood Detector",
        description=(
            "Use your webcam to detect your pet's mood in real-time. "
            "Make sure your pet's face is clearly visible."
        ),
        allow_flagging="never"
    )
    
    return webcam_interface


def main():
    """Main function to run the Gradio app."""
    parser = argparse.ArgumentParser(description="Pet Mood Detector Gradio App")
    parser.add_argument('--model', type=str, default='../models/final_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the Gradio app on')
    parser.add_argument('--share', action='store_true',
                        help='Create a public link for the app')
    args = parser.parse_args()
    
    # Create both interfaces
    image_interface = setup_interface(args.model)
    webcam_interface = setup_webcam_interface(args.model)
    
    # Create a combined interface
    demo = gr.TabbedInterface(
        [image_interface, webcam_interface],
        ["Upload Image", "Use Webcam"],
        title="Pet Mood Detector",
        theme=gr.themes.Soft()
    )
    
    # Launch the app
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
