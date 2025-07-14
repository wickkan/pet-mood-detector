# 🐱🐶 Pet Mood Detector

## Overview

The Pet Mood Detector is a computer vision application that uses deep learning to identify the emotional state of cats and dogs from images or webcam feeds. The model can classify pet moods into four categories: happy, angry, sad, and other/neutral.

## Features

- **Real-time mood detection** via webcam
- **Image upload** for analyzing saved photos
- **Trained on diverse pet expressions**
- **Visual confidence scores** for each emotion category
- **Web interface** using Gradio for easy interaction

## Tech Stack

- Python 3.10+
- PyTorch & torchvision (Deep Learning)
- OpenCV (Image Processing)
- Gradio (Web UI)
- TensorBoard (Training Visualization)
- Matplotlib (Data Visualization)

## Project Structure

```
pet-mood-detector/
│
├── data/                # Dataset organization
│   ├── angry/          # Raw angry pet images
│   ├── happy/          # Raw happy pet images
│   ├── other/          # Raw neutral/other pet images
│   ├── sad/            # Raw sad pet images
│   └── master_folder/  # Organized train/valid/test split
│
├── models/             # Saved model checkpoints
│
├── notebooks/
│   └── 01_train.ipynb  # Interactive training notebook
│
├── src/
│   ├── __init__.py
│   ├── datamodule.py   # Data loading and preprocessing
│   ├── model.py        # Neural network architecture
│   ├── train.py        # Training script
│   └── predict.py      # Inference utility
│
├── ui/
│   └── app.py          # Gradio web interface
│
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/pet-mood-detector.git
   cd pet-mood-detector
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The model is trained on the [Pets Facial Expression Dataset](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset) from Kaggle, which contains pet images categorized by emotional expressions:

- **Happy**: Pets showing signs of joy, play, or excitement
- **Angry**: Pets showing aggression, fear, or discomfort
- **Sad**: Pets showing signs of sadness or illness
- **Other/Neutral**: Default category for neutral or unclear expressions

The data is organized in the `data/master_folder/` directory with standard train/valid/test splits. The dataset was preprocessed and split into training, validation, and test sets to ensure robust model evaluation.

## Training

Train the model using the provided script:

```bash
python src/train.py --data_dir data/master_folder --backbone resnet18 --num_epochs 15
```

Options:

- `--data_dir`: Dataset root directory
- `--backbone`: Model backbone architecture (`resnet18` or `mobilenet_v2`)
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Initial learning rate (default: 0.001)

Alternatively, you can use the Jupyter notebook for an interactive training experience:

```bash
jupyter notebook notebooks/01_train.ipynb
```

## Inference

Run predictions on single images or via webcam:

```bash
# Predict from an image
python src/predict.py --image path/to/pet_image.jpg

# Use webcam for real-time prediction
python src/predict.py --webcam
```

## Web Interface

Launch the Gradio web interface for interactive usage:

```bash
python ui/app.py
```

This opens a browser window where you can:

- Upload pet images for mood detection
- Use your webcam for real-time analysis
- View confidence scores for each emotion category

## Performance

The model achieves approximately 85-90% accuracy on the validation set after training on ResNet18 with transfer learning. Performance may vary depending on image quality, lighting, and whether the pet's face is clearly visible.

## Future Improvements

- Expand emotion categories (e.g., fear, surprise)
- Support for video input with temporal analysis
- Face detection to improve focus on pet facial features
- Mobile app deployment
- Multi-animal detection in a single frame
- Breed-specific emotion models

## Acknowledgements

- Transfer learning uses models pre-trained on ImageNet
- Special thanks to pet owners who contributed to the dataset
