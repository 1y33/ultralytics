import torch
from ultralytics import YOLO

def compare_yolo_output_shapes(cfg_path_a: str,
                               cfg_path_b: str,
                               img_size: int = 640,
                               device: str = "cpu"):
    """
    Instantiate two YOLO models from YAML files, run a dummy forward pass,
    and report the raw output-tensor shapes.

    Args
    ----
    cfg_path_a, cfg_path_b : str
        Paths to the .yaml model definitions for the teacher and student.
    img_size : int, default 640
        Height/width of the square dummy input (pixels).
    device : {"cpu","cuda"} or "cuda:idx"
        Where to place the models and tensor.
    """
    # --- 1. Build the networks from their YAML blueprints --------------------
    model_a = YOLO(cfg_path_a).to(device)
    model_b = YOLO(cfg_path_b).to(device)

    # --- 2. Dummy input -------------------------------------------------------
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)

    # --- 3. Forward pass through the underlying PyTorch models ---------------
    # `YOLO(model_cfg)` wraps an internal `.model` (nn.Module).  Calling it
    # directly gives raw head predictions (no NMS, no post-processing).
    with torch.no_grad():
        pred_a = model_a.model(dummy)[0]   # tensor shape: (B, anchors, 5+nc)d
        pred_b = model_b.model(dummy)[0]

    # --- 4. Report ------------------------------------------------------------
    print(f"Teacher ({cfg_path_a}) output shape: {tuple(pred_a.shape)}")
    print(f"Student ({cfg_path_b}) output shape: {tuple(pred_b.shape)}")

    if pred_a.shape == pred_b.shape:
        print("✅ Shapes match – ready for direct output-level distillation.")
    else:
        print("⚠️ Shapes do NOT match – see suggestions below.")

    # Handy for downstream scripts:
    return pred_a.shape, pred_b.shape

if __name__ == "__main__":
    # Replace with your yaml file paths
    model1_yaml = "yolov5x.yaml"
    model2_yaml = "yolov5n1024.yaml"
    
    # Test with default input size (batch=1, channels=3, height=640, width=640)
    compare_yolo_output_shapes(model1_yaml, model2_yaml,1024)
    
    # test_output_shapes(model1_yaml, model2_yaml, input_size=(1, 3, 416, 416))