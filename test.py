import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Custom objects for deserialization if needed
def custom_objects():
    return {
        'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D  # Ensure using tensorflow.keras
    }

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1)

# Load the model directly using Keras
try:
    model = load_model("model/keras_model.h5", custom_objects=custom_objects())
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define the target image size expected by the model
img_size = 224  # Resize to 224x224 as expected by the model
labels = ["A", "B", "C"]  # Labels for 'A', 'B', and 'C'

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break  # Exit loop if the camera frame cannot be read

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

        h_img, w_img, _ = img.shape
        x1, y1 = max(0, x - 20), max(0, y - 20)
        x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)

        if x2 > x1 and y2 > y1:
            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            aspect_ratio_crop = imgCropShape[1] / imgCropShape[0]
            if aspect_ratio_crop > 1:
                scale_width = img_size
                scale_height = int(img_size / aspect_ratio_crop)
            else:
                scale_height = img_size
                scale_width = int(img_size * aspect_ratio_crop)

            img_resized = cv2.resize(imgCrop, (scale_width, scale_height))

            y_offset = (img_size - scale_height) // 2
            x_offset = (img_size - scale_width) // 2
            img_white[y_offset:y_offset + scale_height, x_offset:x_offset + scale_width] = img_resized

            # Resize to 224x224 to match the model input shape
            img_resized_for_model = cv2.resize(img_white, (224, 224))

            # Convert image to RGB and normalize
            img_processed = cv2.cvtColor(img_resized_for_model, cv2.COLOR_BGR2RGB)
            img_processed = np.expand_dims(img_processed, axis=0)  # Add batch dimension
            img_processed = img_processed / 255.0  # Normalize to 0-1

            # Get the prediction from the model
            try:
                prediction = model.predict(img_processed)
                index = np.argmax(prediction)  # Get the index of the highest probability
                predicted_label = labels[index]
                print(f"Predicted Label: {predicted_label}")

                # Display the prediction label on the image
                cv2.putText(img, f"Predicted: {predicted_label}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error during prediction: {e}")

            # Make sure to display the modified image (with prediction text) in the correct window
            cv2.imshow('Image', img)

    # Display original frame (without text)
    cv2.imshow("Original Image", img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


