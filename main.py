import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the camera feed and HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=4)

# Padding to extend the crop area around the detected hand
padding = 20

while True:
    # Capture a frame from the camera feed
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Detect hands and retrieve frame with annotations
    hands, annotated_frame = detector.findHands(frame, flipType=False)

    if hands:
        # Get details of the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Calculate the dimensions for cropping
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)

        # Crop the image and display if dimensions are valid
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            cv2.imshow('Cropped Image', crop)

    # Display the annotated frame with hand detection results
    cv2.imshow("Frame with Annotations", annotated_frame)

    # Exit the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera feed and close windows
cap.release()
cv2.destroyAllWindows()
