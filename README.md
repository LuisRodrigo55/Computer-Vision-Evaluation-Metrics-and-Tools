
## Quantitative Metrics: The Mathematical Ground Truth

In computer vision, quantitative metrics provide an objective, mathematical evaluation of a model's performance. However, different tasks require different evaluation strategies. A metric that works perfectly for classifying an entire image will fail when evaluating how well a model draws a bounding box around a specific object.

Below, I break down the foundational metrics used for Object Detection and Image Segmentation:

---

### 1. Object Detection Metrics

Object detection models must accomplish two tasks simultaneously: **localization** (drawing a bounding box in the correct place) and **classification** (assigning the correct label to that box).

#### Intersection over Union (IoU)

IoU is the foundational metric for measuring localization accuracy. It calculates the area of overlap between the predicted bounding box and the ground truth bounding box.

$$IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

* **Interpretation:** An IoU of $1$ represents perfect overlap, while $0$ means no overlap.
* **Thresholding:** In practice, an IoU threshold (commonly $0.5$) is set. If the IoU is greater than the threshold, the prediction is considered a **True Positive (TP)**. If it falls below, it is a **False Positive (FP)**.

#### Precision and Recall

Before calculating the overarching metric, we must define precision and recall.

* **Precision:** Out of all the bounding boxes the model predicted as positive, how many were actually correct?

$$Precision = \frac{TP}{TP + FP}$$


* **Recall:** Out of all the actual ground truth objects in the image, how many did the model successfully find?

$$Recall = \frac{TP}{TP + FN}$$



#### Mean Average Precision (mAP)

mAP is the gold standard for object detection evaluation. It provides a single number summarizing the model's performance across all classes.

1. **Average Precision (AP):** For a single class, we plot Precision against Recall at varying confidence thresholds to create a Precision-Recall (PR) curve. AP is the area under this curve.

$$AP = \int_{0}^{1} p(r) dr$$


2. **mAP Calculation:** We calculate the mean of the AP values across all $N$ object classes.

$$mAP = \frac{1}{N} \sum_{i=1}^{N} AP_i$$



* **COCO vs. Pascal VOC:** Note that different datasets define mAP slightly differently. Pascal VOC typically calculates mAP at a single IoU threshold (e.g., $0.5$, denoted as mAP@.50). The COCO dataset calculates mAP averaged over 10 different IoU thresholds (from $0.50$ to $0.95$ in steps of $0.05$) to reward models with tighter localization.

---

### 2. Image Segmentation Metrics

Segmentation requires pixel-perfect accuracy. It is divided into Semantic Segmentation (classifying every pixel into a category) and Instance Segmentation (identifying individual objects at the pixel level).

#### Semantic Segmentation: Mean Intersection over Union (mIoU)

While simple Pixel Accuracy exists, it is easily skewed by background dominance. mIoU is the standard. It applies the IoU concept at the pixel level, calculating the overlap between the predicted segmentation mask and the ground truth mask for a specific class.

$$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$

The final mIoU is the average of these IoU values across all classes ($N_c$).

$$mIoU = \frac{1}{N_c} \sum_{c=1}^{N_c} IoU_c$$

#### Semantic Segmentation: Dice Coefficient (F1 Score)

The Dice Coefficient is mathematically equivalent to the F1 score for segmentation masks and is heavily used in medical imaging. It places more weight on the True Positives than the standard IoU.

$$Dice = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

#### Instance & Panoptic Segmentation Metrics

Instance segmentation combines the localization of object detection with the pixel-level classification of semantic segmentation.

* **Mask AP:** Similar to standard AP used in object detection, but the IoU is calculated based on the overlapping area of the predicted masks rather than bounding boxes.
* **Panoptic Quality (PQ):** Used for Panoptic Segmentation (which unifies semantic and instance segmentation). It multiplies Segmentation Quality (SQ) by Recognition Quality (RQ).

---
