from ultralytics import YOLO

MODEL_CONFIG = "yolov5n1024.yaml"
TRAINING_CONFIG = {
    "data": "coco.yaml",
    "resume": False,
    "batch": 64,
    "epochs": 100,
    "dropout": 0.0,
    "lr0": 1e-3,
    "lrf": 1e-5,
    "name": "Test",
    "imgsz": 256,
    "cos_lr": True,
    "seed": 0, 
    "optimizer": "adam",
    "save_period": 10,
    "cache": False,
    "workers": 8
}


model = YOLO(MODEL_CONFIG).load("yolov5nu.pt")

model.train(**TRAINING_CONFIG)