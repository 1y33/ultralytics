from ultralytics import YOLO
import torch
import ultralytics
import ultralytics.utils
import ultralytics.utils.torch_utils


MODEL_CONFIG = "YOLO/yolov5C2C2fCBAMGhostBig3.yaml"
MODEL_WEIGHTS = "yolov5su.pt"
TRAINED_MODEL_PATH = "runs/detect/train519/weights/best.pt"

TRAINING_CONFIG = {
    "data": "VisDrone.yaml",
    "resume": False,
    "batch": 0.60,
    "epochs": 50,
    "dropout": 0.0,
    "lr0": 1e-3,
    "lrf": 1e-3,
    "name": "Test",
    "imgsz": 1024,
    "cos_lr": True,
    "seed": 0,
    "optimizer": "adam",
    "save_period": 10,
    "cache": False,
    "workers": 8
}

EXPORT_CONFIG = {
    "format": "edgetpu",
    "imgsz": 1024
}

INFERENCE_CONFIG = {
    "test_image": "images/test_image.jpeg",
    "input_shape": [1, 3, 1024, 1024]
}


def train(use_pretrained=False, config_override=None):
    print("Starting training...")
    
    if use_pretrained:
        model = YOLO(MODEL_CONFIG).load(MODEL_WEIGHTS)
        print(f"Loaded model from {MODEL_CONFIG} with weights {MODEL_WEIGHTS}")
    else:
        model = YOLO(MODEL_CONFIG)
        print(f"Loaded model from {MODEL_CONFIG}")
    
    train_config = TRAINING_CONFIG.copy()
    if config_override:
        train_config.update(config_override)
    
    print(f"Training config: {train_config}")
    model.train(**train_config)
    print("Training completed!")
    
    return model

def export(model_path=None, config_override=None):
    print("Starting export...")
    
    path = model_path or TRAINED_MODEL_PATH
    model = YOLO(path)
    print(f"Loaded trained model from {path}")
    
    export_config = EXPORT_CONFIG.copy()
    if config_override:
        export_config.update(config_override)
    
    print(f"Export config: {export_config}")
    exported_path = model.export(**export_config)
    print(f"Model exported to: {exported_path}")
    
    return exported_path

def inference(model_path=None, image_path=None, tensor_inference=True):
    print("Starting inference...")
    
    path = model_path or TRAINED_MODEL_PATH
    model = YOLO(path)
    print(f"Loaded model from {path}")
    
    if image_path:
        print(f"Running inference on image: {image_path}")
        results = model(image_path)
        print("Image inference completed!")
        return results
    
    if tensor_inference:
        print("Running tensor inference...")
        x_input = torch.rand(INFERENCE_CONFIG["input_shape"])
        nnmodel = model.model
        output = nnmodel._predict_once(x_input)
        print(f"Tensor inference completed! Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        return output
    
    print("No inference type specified")
    return None


def main():
    # 1. TRAINING
    # train(use_pretrained=dTrue)  # Train with pre-trained weights
    train(use_pretrained=False)  # Train from scratch
    
    # 2. EXPORT
    # export(MODEL_CONFIG) # Export with default config
    # export(config_override={"format": "onnx", "imgsz": 640})  # Custom export
    
    # 3. INFERENCE
    # inference(image_path=INFERENCE_CONFIG["test_image"])  # Image inference
    # inference(model_path=MODEL_CONFIG,tensor_inference=True)  # Tensor inference
    
    print("Workflow completed!")

if __name__ == "__main__":
    main()