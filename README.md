# YOLOv5 Object Detection Project

This project demonstrates the use of the YOLOv5 model for object detection using Python. It allows you to detect objects in images, videos, or even real-time using a webcam.

## Requirements

To run this project, you need to install the dependencies listed in `requirements.txt`:

```
pip install -r requirements.txt
```

## How to Run

### Detect Objects in an Image or Video
To detect objects in an image or video, use:

```
python detect.py --source <path/to/image_or_video>
```

### Example
Run the detection on the provided test image:

```
python detect.py --source images/bus.jpg
```

## Pre-trained Weights

The YOLOv5 model weights file (`yolov5s.pt`) is automatically downloaded the first time you run the script.

## License

MIT License.