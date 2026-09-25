from pathlib import Path
import csv

from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_FOLDER / "intersection_images"
OUTPUT_FOLDER = BASE_FOLDER / "nms_outputs"
SUMMARY_FILE = OUTPUT_FOLDER / "nms_summary.csv"


# --------------------------------------------------
# Detection settings
# --------------------------------------------------
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 640

# Relevant COCO class IDs:
# 0 = person
# 1 = bicycle
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
TRAFFIC_CLASS_IDS = [0, 1, 2, 3, 5, 7]

# Only the NMS IoU threshold changes
NMS_THRESHOLDS = {
    "small_iou_010": 0.10,
    "normal_iou_045": 0.45,
    "large_iou_095": 0.95
}


# --------------------------------------------------
# Check input folder
# --------------------------------------------------
if not INPUT_FOLDER.exists():
    raise FileNotFoundError(
        f"Could not find the image folder: {INPUT_FOLDER}"
    )


# Create the main output folder
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load pretrained YOLO model
# --------------------------------------------------
model = YOLO("yolo26n.pt")


# --------------------------------------------------
# Run the NMS experiment
# --------------------------------------------------
experiment_results = []

for run_name, nms_threshold in NMS_THRESHOLDS.items():
    print("\n" + "=" * 60)
    print(f"Running NMS experiment with IoU = {nms_threshold:.2f}")
    print("=" * 60)

    results = model.predict(
        source=str(INPUT_FOLDER),

        # Keep the confidence threshold constant
        conf=CONFIDENCE_THRESHOLD,

        # Change only the NMS IoU threshold
        iou=nms_threshold,

        imgsz=IMAGE_SIZE,
        classes=TRAFFIC_CLASS_IDS,

        # Draw and save the results
        save=True,
        show_boxes=True,
        show_labels=True,
        show_conf=True,
        line_width=2,

        project=str(OUTPUT_FOLDER),
        name=run_name,
        exist_ok=True
    )

    # Count detections across the entire dataset
    total_detections = sum(
        len(result.boxes) for result in results
    )

    processed_images = len(results)

    experiment_results.append({
        "run_name": run_name,
        "nms_threshold": nms_threshold,
        "processed_images": processed_images,
        "total_detections": total_detections
    })

    print(f"\nNMS IoU threshold: {nms_threshold:.2f}")
    print(f"Images processed: {processed_images}")
    print(f"Total detections: {total_detections}")
    print(f"Results saved to: {OUTPUT_FOLDER / run_name}")


# --------------------------------------------------
# Save experiment summary to CSV
# --------------------------------------------------
with SUMMARY_FILE.open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "run_name",
        "nms_iou_threshold",
        "processed_images",
        "total_detections"
    ])

    for experiment in experiment_results:
        writer.writerow([
            experiment["run_name"],
            experiment["nms_threshold"],
            experiment["processed_images"],
            experiment["total_detections"]
        ])


# --------------------------------------------------
# Print final comparison
# --------------------------------------------------
print("\n" + "=" * 60)
print("NMS EXPERIMENT SUMMARY")
print("=" * 60)

for experiment in experiment_results:
    print(
        f"IoU {experiment['nms_threshold']:.2f}: "
        f"{experiment['total_detections']} detections "
        f"across {experiment['processed_images']} images"
    )

print(f"\nSummary CSV saved to: {SUMMARY_FILE}")
print(f"Annotated images saved inside: {OUTPUT_FOLDER}")