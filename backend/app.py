from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import base64
import sys
import os

# Add parent directory to path to import from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.emotion_detector import EmotionDetector

app = Flask(__name__)

# Initialize emotion detector (loads lda_model.pkl and svm_model.pkl by default)
emotion_detector = EmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_emotions():
    # Get image data from request
    from flask import request
    
    # Check if image file exists in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read image file directly
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process frame with emotion detector
    processed_frame, result = emotion_detector.detect_emotions(frame)
    
    # Encode processed frame as base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')
    
    # Return processed image and emotion data
    return jsonify({
        'image': f'data:image/jpeg;base64,{processed_image}',
        'emotions': result['emotions']
    })

if __name__ == '__main__':
    app.run(debug=True)
