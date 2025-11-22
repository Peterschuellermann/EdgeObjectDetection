# scripts/export_to_onnx.py

from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"   # <-- your real .pt
IMG_SIZE = 640                   # or whatever you trained with

def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Exporting to ONNX (safe settings)...")
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,     # disable dynamic shapes
        simplify=False,    # disable ONNX simplifier
        opset=12,          # conservative opset
        half=False,        # stay in full float32
        device="cpu",
        verbose=True,
    )

    print("✅ Export complete")

if __name__ == "__main__":
    main()
