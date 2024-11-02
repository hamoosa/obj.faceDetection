import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Forward pass to get output
    outputs = net.forward(output_layers)

    # Process the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Get scores for each class
            class_id = np.argmax(scores)  # Get the index of the highest score
            confidence = scores[class_id]  # Get the confidence of that class

            # Filter out weak predictions; check for person (class_id 0) and telephone (adjust class_id accordingly)
            if confidence > 0.5:
                # Assuming class_id for telephone is 67 (check coco.names)
                if class_id == 0 or class_id == 67:  # Class ID for 'person' and 'cell phone'
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw bounding box around detected object
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame with rectangles
    cv2.imshow('Face and Object Detection with YOLO', frame)

    # Press 'q' to exit the webcam window
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
