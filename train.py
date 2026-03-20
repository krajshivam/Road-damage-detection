"""
Road Damage Detection — YOLOv8 Training Script
Dataset: RDD2022 v10 (7 classes, 8586 images)
"""

from ultralytics import YOLO


def train():
    # Load YOLOv8 nano model with pretrained COCO weights

    model = YOLO("runs/detect/runs/rdd_baseline2/weights/last.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        device="mps",
        amp=False,  # fix for the crash
        project="runs",
        name="rdd_baseline",
        pretrained=True,
        resume=True,  # resume from epoch 25
        plots=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    train()
