import cv2
import torch

# Load the YOLOv5 model (YOLOv5s is the small model, you can use larger ones for more accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# OpenCV video capture (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert frame to RGB as YOLOv5 expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use YOLOv5 model to detect objects in the frame
    results = model(rgb_frame)

    # Render the results on the frame
    result_frame = results.render()[0]  # Rendered results

    # Display the resulting frame with bounding boxes and labels
    cv2.imshow('YOLO Object Detection', result_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
