import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def calculate_iou(boxA, boxB):
    #Calculates Intersection over Union (IoU) for two bounding boxes
    #Boxes are expected in format: [x_min, y_min, x_max, y_max]

    #Coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    #Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    #Compute area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    #Compute the intersection over union
    #Union = AreaA + AreaB - Intersection
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_map():
    #Mock Predictions, list of dictionaries (one dictionary per image)
    #[x_min, y_min, x_max, y_max]
    preds = [
        dict(
            boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            scores=torch.tensor([0.9]),
            labels=torch.tensor([0]),
        )
    ]
    
    target = [
        dict(
            boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            labels=torch.tensor([0]),
        )
    ]
    
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    metric.update(preds, target)
    
    results = metric.compute()    
    return results

def evaluate_detection():
    print("Object Detection Metrics")
    
    gt_box = [50, 50, 150, 150]
    pred_box = [60, 60, 160, 160]
    iou_score = calculate_iou(gt_box, pred_box)
    print(f"Manual IoU Calculation:")
    print(f"Ground Truth: {gt_box}")
    print(f"Prediction  : {pred_box}")
    print(f"IoU Score   : {iou_score:.4f}\n")
    
    print("TorchMetrics mAP Calculation (COCO format):")
    map_results = evaluate_map()
    print(f"mAP (IoU 0.50:0.95): {map_results['map'].item():.4f}")
    print(f"mAP@0.50         : {map_results['map_50'].item():.4f}")
    print(f"mAP@0.75         : {map_results['map_75'].item():.4f}")

if __name__ == "__main__":
    evaluate_detection()
