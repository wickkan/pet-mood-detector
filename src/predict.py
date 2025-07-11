"""
Prediction script for Pet Mood Detector.
"""
import os
import argparse
import time
from typing import Tuple, List, Dict, Union

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import PetMoodClassifier


def load_model(model_path: str, device: str = 'cpu', backbone="mobilenet_v3_small", lightweight=True):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        backbone: Model backbone to use
        lightweight: Whether to use lightweight model architecture
        
    Returns:
        Loaded model
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = PetMoodClassifier(backbone=backbone, lightweight=lightweight)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path} using {backbone} backbone")
    return model


def get_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get transformation for inference.
    
    Args:
        image_size: Target image size
        
    Returns:
        Image transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(
    image_path: str,
    model: PetMoodClassifier,
    device: str = 'cpu',
    image_size: int = 224,
    class_names: List[str] = None
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Predict mood from an image file.
    
    Args:
        image_path: Path to the image file
        model: Trained model
        device: Device to run inference on
        image_size: Target image size
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    if class_names is None:
        class_names = ['angry', 'happy', 'other', 'sad']
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    # Get transformation
    transform = get_transform(image_size)
    
    # Apply transformation and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]
        
        # Get all probabilities as a dict
        all_probs = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    # Convert image for visualization
    img_array = np.array(image)
    
    return {
        "mood": predicted_class,
        "confidence": confidence.item(),
        "probabilities": all_probs,
        "image": img_array
    }


def predict_frame(
    frame: np.ndarray,
    model: PetMoodClassifier,
    device: str = 'cpu',
    image_size: int = 224,
    class_names: List[str] = None
) -> Dict[str, Union[str, float]]:
    """
    Predict mood from a video frame.
    
    Args:
        frame: Input frame (BGR format from OpenCV)
        model: Trained model
        device: Device to run inference on
        image_size: Target image size
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    if class_names is None:
        class_names = ['angry', 'happy', 'other', 'sad']
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Get transformation
    transform = get_transform(image_size)
    
    # Apply transformation and add batch dimension
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_idx.item()]
        
        # Get all probabilities
        all_probs = {class_names[i]: prob.item() for i, prob in enumerate(probabilities)}
    
    return {
        "mood": predicted_class,
        "confidence": confidence.item(),
        "probabilities": all_probs
    }


def webcam_inference(
    model: PetMoodClassifier,
    device: str = 'cpu',
    camera_id: int = 0,
    class_names: List[str] = None
) -> None:
    """
    Run real-time inference on webcam feed.
    
    Args:
        model: Trained model
        device: Device to run inference on
        camera_id: Camera ID (usually 0 for built-in webcam)
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['angry', 'happy', 'other', 'sad']
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting webcam feed... Press 'q' to quit.")
    
    # Color mapping for different moods (BGR format)
    color_map = {
        'happy': (0, 255, 0),    # Green
        'angry': (0, 0, 255),    # Red
        'sad': (255, 0, 0),      # Blue
        'other': (255, 255, 0)   # Cyan
    }
    
    # Inference loop
    frame_count = 0
    skip_frames = 2  # Process every nth frame to improve performance
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Process every nth frame
        frame_count += 1
        if frame_count % skip_frames != 0:
            # Just display the frame without prediction
            cv2.imshow('Pet Mood Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Make prediction
        result = predict_frame(frame, model, device, class_names=class_names)
        mood = result['mood']
        confidence = result['confidence']
        
        # Get color based on predicted mood
        color = color_map.get(mood.lower(), (255, 255, 255))
        
        # Display prediction on frame
        text = f"{mood}: {confidence:.2f}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        
        # Show the frame
        cv2.imshow('Pet Mood Detector', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def visualize_prediction(result: Dict) -> None:
    """
    Visualize image prediction with result overlay.
    
    Args:
        result: Prediction result dictionary
    """
    # Get data from result
    image = result['image']
    mood = result['mood']
    confidence = result['confidence']
    probabilities = result['probabilities']
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {mood} ({confidence:.2f})")
    plt.axis('off')
    
    # Display bar chart of probabilities
    plt.subplot(1, 2, 2)
    moods = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ['red' if mood == m else 'gray' for m in moods]
    
    bars = plt.bar(range(len(moods)), values, color=colors)
    plt.xticks(range(len(moods)), moods, rotation=45)
    plt.ylabel('Probability')
    plt.title('Class Probabilities')
    
    # Add probability values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.show()


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Pet Mood Detection")
    parser.add_argument('--model', type=str, default='models/final_model.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to the image for prediction')
    parser.add_argument('--webcam', action='store_true',
                        help='Use webcam for real-time prediction')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID for webcam capture')
    parser.add_argument('--device', type=str, default='',
                        help="Device to run on ('cuda' or 'cpu')")
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                        choices=['resnet18', 'mobilenet_v2', 'mobilenet_v3_small'],
                        help='Model backbone architecture')
    parser.add_argument('--lightweight', action='store_true', default=True,
                        help='Use lightweight model architecture to reduce model size')
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    model = load_model(args.model, device, backbone=args.backbone, lightweight=args.lightweight)
    
    # Determine class names
    class_names = ['angry', 'happy', 'other', 'sad']  # Default class names
    
    if args.webcam:
        # Run webcam inference
        webcam_inference(model, device, args.camera_id, class_names)
    elif args.image:
        # Run single image prediction
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            return
        
        # Predict and visualize
        result = predict_image(args.image, model, device, class_names=class_names)
        print(f"Prediction: {result['mood']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Visualize
        visualize_prediction(result)
    else:
        print("Error: Either --image or --webcam must be specified")


if __name__ == "__main__":
    main()
