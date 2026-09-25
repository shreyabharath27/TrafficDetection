import csv
from pathlib import Path

# --------------------------------------------------
# File locations
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "outputs" / "task4_results" / "class_counts.csv"
OUTPUT_CSV = PROJECT_DIR / "outputs" / "task4_results" / "safety_analysis.csv"

analysis_rows = []

# --------------------------------------------------
# Read detection counts
# --------------------------------------------------
with INPUT_CSV.open("r", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        image_name = row["image"]

        persons = int(row["person"])
        bicycles = int(row["bicycle"])
        cars = int(row["car"])
        buses = int(row["bus"])
        trucks = int(row["truck"])

        # Motorcycles are excluded because none were detected.
        vehicle_count = cars + buses + trucks

        # This represents vulnerable-road-user activity, not necessarily
        # the exact number of unique people.
        vulnerable_activity = persons + bicycles

        # Higher when vehicles and vulnerable users appear together.
        exposure_score = vehicle_count * vulnerable_activity

        mixed_traffic = (
            "Yes"
            if vehicle_count > 0 and vulnerable_activity > 0
            else "No"
        )

        analysis_rows.append({
            "image": image_name,
            "persons": persons,
            "bicycles": bicycles,
            "cars": cars,
            "buses": buses,
            "trucks": trucks,
            "total_vehicles": vehicle_count,
            "vulnerable_activity": vulnerable_activity,
            "mixed_traffic": mixed_traffic,
            "exposure_score": exposure_score
        })

# --------------------------------------------------
# Rank images from highest to lowest exposure
# --------------------------------------------------
analysis_rows.sort(
    key=lambda row: row["exposure_score"],
    reverse=True
)

# --------------------------------------------------
# Save application-level results
# --------------------------------------------------
fieldnames = [
    "image",
    "persons",
    "bicycles",
    "cars",
    "buses",
    "trucks",
    "total_vehicles",
    "vulnerable_activity",
    "mixed_traffic",
    "exposure_score"
]

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(analysis_rows)

# --------------------------------------------------
# Dataset summary
# --------------------------------------------------
total_images = len(analysis_rows)

mixed_traffic_images = sum(
    1 for row in analysis_rows
    if row["mixed_traffic"] == "Yes"
)

mixed_percentage = (
    (mixed_traffic_images / total_images) * 100
    if total_images > 0
    else 0
)

print("\nINTERSECTION SAFETY ANALYSIS")
print("----------------------------------------")
print(f"Images analyzed: {total_images}")
print(f"Images containing both groups: {mixed_traffic_images}")
print(f"Mixed-traffic percentage: {mixed_percentage:.1f}%")

print("\nTop 5 scenes by potential conflict exposure:")

for rank, row in enumerate(analysis_rows[:5], start=1):
    print(
        f"{rank}. {row['image']} | "
        f"vehicles={row['total_vehicles']}, "
        f"persons={row['persons']}, "
        f"bicycles={row['bicycles']}, "
        f"score={row['exposure_score']}"
    )

print(f"\nFull results saved to:\n{OUTPUT_CSV}")