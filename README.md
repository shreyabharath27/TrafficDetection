# Intersection Safety Traffic Detector

This project uses a pretrained Ultralytics YOLO object detector to analyze traffic scenes at urban intersections. The detector identifies road users, draws bounding boxes, displays class labels and confidence scores, and counts detected objects.

The project focuses on interactions between vehicles and vulnerable road users such as pedestrians and cyclists.

## Detected Classes

- Person
- Bicycle
- Car
- Bus
- Truck

## Dataset

The test dataset contains 24 publicly available intersection images with a variety of:

- Lighting and weather conditions
- Camera viewpoints
- Object sizes and distances
- Crowded and uncrowded scenes
- Partially occluded objects

## Project Features

- Processes multiple traffic images with a pretrained YOLO model
- Draws bounding boxes around detected objects
- Displays object classes and confidence scores
- Counts relevant object classes in each image
- Saves annotated images and detection results
- Compares confidence thresholds of 0.20, 0.50, and 0.80
- Compares multiple Non-Maximum Suppression IoU thresholds
- Produces an application-level intersection safety analysis

## Repository Files

| File or folder | Description |
| --- | --- |
| `intersection_images/` | Original test images |
| `outputs/` | Annotated images, labels, and CSV results |
| `nms_outputs/` | Images and results from the NMS experiments |
| `detect.py` | Runs detection, draws boxes, and counts objects |
| `nms_experiments.py` | Tests different NMS IoU thresholds |
| `application_analysis.py` | Calculates the potential conflict-exposure score |
| `yolo26n.pt` | Pretrained YOLO model weights |

## Installation

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy
```

## Running the Project

Run the main detector:

```bash
python detect.py
```

Run the NMS experiments:

```bash
python nms_experiments.py
```

Run the application-level analysis:

```bash
python application_analysis.py
```

Generated images and detection data are saved in the `outputs/` and `nms_outputs/` folders.

## Application-Level Analysis

The analysis groups detections into two categories:

- **Vehicles:** cars, trucks, and buses
- **Vulnerable-road-user detections:** people and bicycles

The potential conflict-exposure score is calculated as:

```text
vehicles x (people + bicycles)
```

A higher score means that more vehicles and vulnerable-road-user detections appear in the same image. The score identifies scenes that may deserve additional safety monitoring, but it does not predict accidents.

## Limitations

The detector may miss objects that are small, distant, crowded, or partially occluded. It may also confuse visually similar classes, such as cars and trucks. A cyclist can produce both a person detection and a bicycle detection, so the application score should not be interpreted as the number of unique individuals.
