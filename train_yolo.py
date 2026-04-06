import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


def main():

    os.chdir(r"C:\Users\aadri\OneDrive\Escritorio\rec_ob")

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        augment=True
    )


if __name__ == "__main__":
    main()