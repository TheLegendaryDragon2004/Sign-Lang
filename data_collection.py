import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
img_size = 300
counter = 0;
folder = "data/C";
while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if the camera frame cannot be read

    # Detect hands in the image
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

        # Ensure cropping coordinates are within the image bounds
        h_img, w_img, _ = img.shape
        x1, y1 = max(0, x - 20), max(0, y - 20)
        x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)

        # Check if the crop area is valid
        if x2 > x1 and y2 > y1:
            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            # Resize imgCrop to fit into img_white
            aspect_ratio_crop = imgCropShape[1] / imgCropShape[0]
            if aspect_ratio_crop > 1:
                # Width is greater
                scale_width = img_size
                scale_height = int(img_size / aspect_ratio_crop)
            else:
                # Height is greater
                scale_height = img_size
                scale_width = int(img_size * aspect_ratio_crop)

            img_resized = cv2.resize(imgCrop, (scale_width, scale_height))

            # Center the resized crop in the white canvas
            y_offset = (img_size - scale_height) // 2
            x_offset = (img_size - scale_width) // 2
            img_white[y_offset:y_offset + scale_height, x_offset:x_offset + scale_width] = img_resized

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', img_white)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',img_white)
        print(counter)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
