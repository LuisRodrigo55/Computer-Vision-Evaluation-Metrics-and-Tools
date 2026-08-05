import numpy as np

def GenerateMask(size=(256, 256)):
   #Generate binary masks for a single class 1 is the object and 0 the background
    
    y_true = np.zeros(size, dtype=np.uint8)
    y_true[100:150, 100:150] = 1
    
    # Create a prediction mask
    y_pred = np.zeros(size, dtype=np.uint8)
    y_pred[110:160, 90:140] = 1
    
    return y_true, y_pred

def CalcMIOU(y_true, y_pred, num_classes=2): #Mean Intersection over Union (mIoU)

  ious = []
    for c in range(num_classes):
        # Extract binary masks for the current class
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        
        # Calculate intersection and union
        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()
        
        # Avoid division by zero if the class is completely absent
        if union == 0:
            ious.append(float('nan'))  # Ignore this class
        else:
            ious.append(intersection / union)
            
    # Return the mean of valid IoUs
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious)

def CalculateDice(y_true, y_pred): #Calculates the Dice Coefficient (F1 Score) for a binary mask
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    # Calculate intersection
    intersection = np.sum(y_true_f * y_pred_f)
    
    # Dice Formula: (2 * TP) / (2 * TP + FP + FN)
    # (2 * Intersection) / (Total pixels in true + Total pixels in pred)
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
    return dice

def EvaluateSegmentation():
    y_true, y_pred = GenerateMask()
    
    # Calculate metrics
    miou = CalcMIOU(y_true, y_pred, num_classes=2)
    dice = CalculateDice(y_true, y_pred)
    
    print(f"Mean Intersection over Union (mIoU): {miou:.4f}")
    print(f"Dice Coefficient (Binary)          : {dice:.4f}")

if __name__ == "__main__":
    EvaluateSegmentation()
