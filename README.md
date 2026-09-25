Traffic Detection for Intersection Safety

This project uses a pretrained Ultralytics YOLO object-detection model to analyze traffic scenes at urban intersections. It detects relevant road users, including people, bicycles, cars, buses, and trucks, and produces annotated images containing bounding boxes, class labels, and confidence scores.

The goal is to explore how object detection can support pedestrian and cyclist safety analysis. Detection counts are used to identify scenes where vehicles and vulnerable road users appear together and to calculate a simple potential conflict-exposure score.

Project Features

- Processes 24 intersection images with different lighting conditions, viewpoints, traffic densities, object sizes, and levels of occlusion.

- Draws bounding boxes with class labels and confidence scores.

- Counts detections for each relevant road-user class.

- Saves annotated images and CSV files containing detection results.

- Compares confidence thresholds of 0.20, 0.50, and 0.80.

- Experiments with multiple Non-Maximum Suppression (NMS) IoU thresholds.

- Calculates an application-level potential conflict-exposure score.

**Application-Level Analysis
**
For each image, the program groups detections as follows:

Vehicles: cars, trucks, and buses

Vulnerable-road-user detections: people and bicycles

The potential conflict-exposure score is calculated as:

vehicles * (people + bicycles)

A higher score indicates that more vehicle and vulnerable-road-user detections appear in the same scene. This score measures potential exposure only; it does not predict accidents because the detector does not estimate speed, direction, or real-world distance.

Repository Structure

TrafficDetection/
├── intersection_images/      # Input traffic images
├── outputs/                  # Annotated images, labels, and CSV results
├── nms_outputs/              # Results from the NMS experiments
├── detect.py                 # Detection, visualization, and class counting
├── nms_experiments.py        # NMS IoU-threshold experiments
├── application_analysis.py   # Potential conflict-exposure analysis
└── yolo26n.pt                # Pretrained YOLO model weights

Requirements

Python 3.10 or newer

Ultralytics

OpenCV

NumPy

Install the dependencies with:

pip install ultralytics opencv-python numpy

Running the Project

From the repository directory, run:

python detect.py
python nms_experiments.py
python application_analysis.py

The generated annotated images and detection data are saved in the outputs/ and nms_outputs/ directories.

Limitations

The detector may miss small, distant, crowded, or partially occluded objects. It may also confuse visually similar classes, such as cars and trucks. In addition, a cyclist may be represented by both a person detection and a bicycle detection, so the application-level score should not be interpreted as a count of unique individuals.
