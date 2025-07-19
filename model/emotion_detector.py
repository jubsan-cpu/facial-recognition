import cv2
import numpy as np
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# Removed sklearn imports

class EmotionDetector:
    def __init__(self, model_path='model/emotion_model.h5'):
        # Emotion labels MUST match the order from the training generator's class_indices
        # Found during training: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        self.emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self.model = None # Initialize model attribute

        # Load trained Keras CNN model
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Loaded Keras CNN model from {model_path}")
                # Optional: Print model summary
                # self.model.summary()
            except Exception as e:
                print(f"Error loading Keras model: {e}")
                self.model = None
        else:
            print(f"Error: Keras model file {model_path} not found. Predictions will not work.")
            self.model = None

        # Load face detection model (using OpenCV's Haar Cascade)
        haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(haar_cascade_path):
             print(f"Error: Haar Cascade file not found at {haar_cascade_path}")
             self.face_cascade = None
        else:
             self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
             if self.face_cascade.empty():
                  print("Error: Failed to load Haar Cascade classifier.")
                  self.face_cascade = None
             else:
                  print("Loaded face detection model (Haar Cascade).")
    
    def detect_emotions(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = {
            'faces': [],
            'emotions': {}
        }
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 (model input size)
            resized_face = cv2.resize(face_roi, (48, 48))
            
            # Normalize
            normalized_face = resized_face / 255.0
            
            # Store face coordinates
            face_coords = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
            result['faces'].append(face_coords)
            
            # Predict emotions if model and face cascade are loaded
            emotion_scores = {} # Initialize scores dictionary
            dominant_emotion = "N/A" # Default if no prediction

            if self.model is not None and self.face_cascade is not None:
                # Reshape for CNN input: (batch_size, height, width, channels)
                cnn_input = normalized_face.reshape(1, 48, 48, 1)

                # Get CNN prediction probabilities
                try:
                    emotion_probs = self.model.predict(cnn_input, verbose=0)[0] # verbose=0 suppresses progress bar

                    # Convert probabilities to intensity scores (0-100)
                    for i, emotion in enumerate(self.emotions):
                        intensity = int(emotion_probs[i] * 100)
                        emotion_scores[emotion] = intensity

                    # Get dominant emotion
                    dominant_emotion_idx = np.argmax(emotion_probs)
                    if dominant_emotion_idx < len(self.emotions):
                         dominant_emotion = self.emotions[dominant_emotion_idx]
                    else:
                         print(f"Warning: Predicted index {dominant_emotion_idx} out of bounds for emotions list.")
                         dominant_emotion = "Error"

                except Exception as e:
                    print(f"Error during model prediction: {e}")
                    dominant_emotion = "Error"
                    # Populate scores with error indication if needed
                    for emotion in self.emotions:
                        emotion_scores[emotion] = -1 # Or some other indicator

            else:
                 print("Warning: Model or face cascade not loaded. Skipping prediction.")
                 # Populate scores with default values if prediction skipped
                 for emotion in self.emotions:
                     emotion_scores[emotion] = 0

            dominant_score = emotion_scores[dominant_emotion]
            
            # Determine color based on emotion valence
            # Green for positive, red for negative, yellow for neutral
            if dominant_emotion in ['Happy', 'Surprise']:
                color = (0, 255, 0)  # Green (BGR)
            elif dominant_emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
                color = (0, 0, 255)  # Red (BGR)
            else:  # Neutral
                color = (0, 255, 255)  # Yellow (BGR)
            
            # Add wireframe with appropriate color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Store emotion scores
            result['emotions'] = emotion_scores
        
        return frame, result
        
    def process_image(self, image_path):
        """
        Process a single image file and return emotion analysis
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary containing emotion scores and face coordinates
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': 'Could not read image file'}
            
        # Process frame
        _, result = self.detect_emotions(frame)
        
        # Save annotated image
        output_path = os.path.join('output', os.path.basename(image_path))
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_path, frame)
        
        return {
            'image_path': output_path,
            'emotions': result['emotions'],
            'faces': result['faces']
        }
