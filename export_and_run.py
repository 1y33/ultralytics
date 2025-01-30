from ultralytics import YOLO
import ultralytics
import ultralytics.utils
import ultralytics.utils.torch_utils

def main():
    # path = "runs/detect/imgsze-640-training--EP:100-BS:16+0.0001/weights/best.pt"
    # m.detect_image("images/test_image.jpeg")
    # model = YOLO("runs/detect/train5/weights/best.pt")
    #model.train(data = "VisDrone.yaml",epochs=1)
    # model.export(format="edgetpu", imgsz=256);
    #model.save(path = "models/visdrone_1_epoch")
    
    
    
    # Load model yaml
    # model_loaded_weights = YOLO("yolov5newtest.yaml").load("yolov5su.pt")
    # model_after_training = YOLO("runs/detect/train519/weights/best.pt")    
    model = YOLO("YOLO/yolov5sC2C3GhostBig2.yaml")
    # model = YOLO('yolov5nu.yaml')
    imgsz = 1024

    model.train(data="VisDrone.yaml",resume=False, batch=.60, epochs=50, dropout=0.0, lr0=1e-3, lrf=1e-3, name="Test", imgsz=imgsz, cos_lr=True, seed=0, optimizer="adam", save_period=10, cache=False, workers=8)
    # import torch
    # from ultralytics.nn.modules.conv import CBAM
    # model.export(format="edgetpu", imgsz=imgsz);


    import torch
    x_input = torch.rand([1,3,1024,1024])
    nnmodel = model.model
    nnmodel._predict_once(x_input)

main()
