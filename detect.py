import torch
import cv2
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='images/bus.jpg', help='Path to source image or video')
parser.add_argument('--weights', type=str, default='yolov5s.pt', help='Path to YOLOv5 weights file')
args = parser.parse_args()

# Load YOLOv5 model from the PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)

# Load an image or video from the specified source
source = args.source
if not os.path.exists(source):
    raise FileNotFoundError(f"Source file '{source}' does not exist")

# Check if source is a video or image
if source.endswith(('.mp4', '.avi')):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)
        
        # Render results on the frame
        results.render()

        # Display the frame with detection results
        cv2.imshow('YOLOv5 Object Detection', results.imgs[0])

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    # If it's an image, read and process the image
    img = cv2.imread(source)
    results = model(img)
    results.render()

    # Save the resulting image to the output directory
    output_path = 'runs/detect/exp'
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, os.path.basename(source))
    cv2.imwrite(output_file, results.ims[0])
    print(f"Results saved to {output_file}")

    # Display results
    results.show()

cv2.destroyAllWindows()
