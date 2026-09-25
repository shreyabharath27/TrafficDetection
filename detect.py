from pathlib import Path
from collections import Counter
import csv

from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_FOLDER / "intersection_images"
OUTPUT_FOLDER = BASE_FOLDER / "outputs"
RUN_NAME = "task4_results"


# --------------------------------------------------
# Detection settings
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.80
IMAGE_SIZE = 640

# Relevant COCO classes:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
TRAFFIC_CLASS_IDS = [0, 1, 2, 3, 5, 7]

TRAFFIC_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
]


# --------------------------------------------------
# Check input folder
# --------------------------------------------------
if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"Could not find image folder: {INPUT_FOLDER}"
    )


# --------------------------------------------------
# Load pretrained model
# --------------------------------------------------
model = YOLO("yolo26n.pt")


# --------------------------------------------------
# Process all images
# --------------------------------------------------
results = model.predict(
    source=str(INPUT_FOLDER),
    conf=CONFIDENCE_THRESHOLD,
    iou=IOU_THRESHOLD,
    imgsz=IMAGE_SIZE,
    classes=TRAFFIC_CLASS_IDS,

    # Save bounding boxes, classes, and confidence scores
    save=True,
    save_txt=True,
    save_conf=True,
    show_labels=True,
    show_conf=True,
    line_width=2,

    project=str(OUTPUT_FOLDER),
    name=RUN_NAME,
    exist_ok=True
)


if not results:
    raise RuntimeError("No images were successfully processed.")


# Use the exact output directory created by YOLO
run_folder = Path(results[0].save_dir)

details_csv_path = run_folder / "detection_details.csv"
counts_csv_path = run_folder / "class_counts.csv"


# --------------------------------------------------
# Save detailed detections and per-image counts
# --------------------------------------------------
with (
    details_csv_path.open("w", newline="") as details_file,
    counts_csv_path.open("w", newline="") as counts_file
):
    details_writer = csv.writer(details_file)
    counts_writer = csv.writer(counts_file)

    details_writer.writerow([
        "image",
        "object_class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    counts_writer.writerow([
        "image",
        *TRAFFIC_CLASS_NAMES,
        "total_objects"
    ])

    for result in results:
        image_name = Path(result.path).name
        class_counts = Counter()

        # Examine every detected object
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Count this object
            class_counts[class_name] += 1

            # Save its detailed information
            details_writer.writerow([
                image_name,
                class_name,
                round(confidence, 4),
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ])

        total_objects = sum(class_counts.values())

        # Save one summary row for this image
        counts_writer.writerow([
            image_name,
            *[
                class_counts[class_name]
                for class_name in TRAFFIC_CLASS_NAMES
            ],
            total_objects
        ])

        # Print counts in the required format
        print(f"\nImage: {image_name}")

        for class_name in TRAFFIC_CLASS_NAMES:
            print(f"{class_name}: {class_counts[class_name]}")

        print(f"Total detected objects: {total_objects}")


print("\nTask 4 processing complete!")
print(f"Annotated images: {run_folder}")
print(f"Class counts: {counts_csv_path}")
print(f"Detection details: {details_csv_path}")