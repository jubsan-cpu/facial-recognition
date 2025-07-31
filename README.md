# Emotion Detection System

A real-time emotion detection system that analyzes facial expressions from images and classifies them into seven distinct emotions: Anger, Disgust, Fear, Happiness, Neutral, Sadness, and Surprise. This project leverages computer vision and deep learning, featuring a retro-styled user interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Details](#model-details)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Project Overview

This project implements a web-based system for detecting emotions in facial images using a custom-trained convolutional neural network (CNN) on the FER-2013 dataset. The user interface mimics a retro terminal with pixelated fonts and green-on-black color schemes.

## Features

- Real-time image upload and emotion recognition
- Seven emotion classes: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
- Retro-styled, responsive web interface
- Colored wireframe overlays on detected faces
- Dynamic emotion intensity visualization
- Graceful error handling and feedback

## System Architecture

```
Frontend (HTML/CSS/JS) <--> Flask Backend (Python) <--> Emotion Model (TensorFlow/Keras)
```

### Data Flow

1. **Image Upload:** User uploads an image via the web UI.
2. **Face Detection:** OpenCV locates faces using Haar Cascade.
3. **Preprocessing:** Detected faces are resized to 48x48, converted to grayscale, and normalized.
4. **Emotion Prediction:** TensorFlow model predicts emotion scores.
5. **Result Display:** Emotion scores and overlays are displayed in the UI.

## Tech Stack

- **Python 3.9+**
- **TensorFlow/Keras** (CNN model for emotion detection)
- **OpenCV (cv2)** (Face detection and image processing)
- **Flask** (Backend server and API)
- **HTML/CSS/JavaScript** (Frontend interface)
- **NumPy** (Array operations)
- **Kaggle API** (Dataset exploration and development)
- **Retro fonts and CSS** for UI styling

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jubsan-cpu/facial-recognition.git
   cd facial-recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download FER-2013 dataset and place model files**
   - Download the FER-2013 dataset from Kaggle.
   - Train the model or use the provided pre-trained weights (if available).

4. **Run the Flask server**
   ```bash
   cd backend
   python app.py
   ```

5. **Start using the web app**
   - Open your browser and go to `http://localhost:*`

## Usage

- Upload a facial image through the UI.
- The system detects the face, analyzes the emotion, and displays the results with a colored wireframe and emotion scores.

## File Structure

```
/model         # ML model and emotion detection logic
/backend       # Flask server and API endpoints
/templates     # HTML templates
/facial examples # Sample images for testing
```

## Model Details

- **Input:** 48x48 grayscale facial images
- **Architecture:** Custom CNN with 3 conv blocks, batch normalization, max pooling, and dropout
- **Training:** FER-2013 dataset, data augmentation, 60 epochs, early stopping, cross-entropy loss
- **Output:** 7 emotion classes (softmax probabilities)
- **Face Detection:** OpenCV Haar Cascade

## Design Decisions

- End-to-end CNN for integrated face detection and emotion classification
- Retro terminal-style UI for unique user experience
- Intuitive color-coded feedback (Green = Positive, Red = Negative, Yellow = Neutral)
- Robust error handling for uploads and processing

## Limitations

- May struggle with non-frontal or poorly lit faces
- Emotion accuracy may vary across demographics
- Processes only one image at a time (no video stream)
- Requires manual file upload (no camera integration)

## Future Enhancements

- Real-time webcam integration
- Multi-face detection and tracking
- Improved accuracy with transfer learning
- Temporal emotion analysis (emotion changes over time)
- User authentication and emotion history

## License

[MIT License](LICENSE)

---

**Contributions are welcome!**
