"""
Road Damage Detection — Evaluation Script
Runs best.pt on test set and generates:
- Final mAP50 and mAP50-95 scores
- Per-class AP breakdown
- Prediction visualizations on sample images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import random


def evaluate():
    model = YOLO("runs/best.pt")

    print("Running evaluation on test set...")
    metrics = model.val(
        data="dataset/data.yaml",
        split="test",
        device="mps",
        plots=True,
        save_json=True,
    )

    print("\n===== FINAL TEST SET RESULTS =====")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n===== PER CLASS AP =====")
    names = [
        "Alligator Cracks",
        "Damaged crosswalk",
        "Damaged paint",
        "Longitudinal Cracks",
        "Manhole cover",
        "Potholes",
        "Transverse Cracks",
    ]
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {names[i]:<25} AP50: {ap:.4f}")


def visualize_predictions(num_images=10):
    """Run inference on random test images and save visualized results."""
    model = YOLO("runs/best.pt")

    test_images = list(Path("dataset/test/images").glob("*.jpg"))
    sample = random.sample(test_images, min(num_images, len(test_images)))

    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating predictions on {len(sample)} sample images...")
    results = model.predict(
        source=sample,
        device="mps",
        conf=0.25,
        save=True,
        project="predictions",
        name="samples",
    )
    print(f"✅ Saved to predictions/samples/")


if __name__ == "__main__":
    evaluate()
    visualize_predictions()
