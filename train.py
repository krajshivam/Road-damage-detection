from ultralytics import YOLO


def train():
    model = YOLO("yolov8s.pt")  # upgraded from nano to small

    model.train(
        data="dataset/data.yaml",
        epochs=100,  # increased from 50
        imgsz=640,
        batch=32,
        device="mps",
        amp=False,
        project="runs/detect/runs",
        name="rdd_v2",  # new name so baseline is preserved
        pretrained=True,
        plots=True,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    train()
