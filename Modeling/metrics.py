import torch
from torchvision.ops import box_iou
from itertools import product


class ConfusionMatrix():
  def __init__(self, num_classes, device):
    self.num_classes = num_classes
    self.confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

  def update(self, targets, outputs):
    n = self.num_classes
    with torch.no_grad():
      mask = (targets >= 0) & (targets < n)
      inds = n * targets[mask].to(torch.int64) + outputs[mask]
      self.confmat += torch.bincount(inds, minlength=n**2).reshape(n, n)

  def confusion_matrix(self, targets, outputs):
    n = self.num_classes
    with torch.no_grad():
      mask = (targets >= 0) & (targets < n)
      inds = n * targets[mask].to(torch.int64) + outputs[mask]
      return torch.bincount(inds, minlength=n**2).reshape(n, n)

  def reduce_from_all_processes(self):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      torch.distributed.barrier()
      torch.distributed.all_reduce(self.confmat)

  def get_accuracy(self):
    """Returns the global accuracy."""
    return self.confmat.diag().sum() / self.confmat.sum()

  def get_row_accuracy(self):
    """Returns the row accuracy."""
    return self.confmat.diag() / self.confmat.sum(0)

  def get_recall(self):
    """Returns the recall."""
    return self.confmat.diag() / self.confmat.sum(1)

  def get_f1_score(self):
    """Returns the F1 score."""
    precision = self.confmat.diag() / self.confmat.sum(0)
    recall = self.confmat.diag() / self.confmat.sum(1)
    return 2 * (precision * recall) / (precision + recall)

  def get_row_iou(self):
    """Returns the row IoU."""
    m = self.confmat.float()
    return m.diag() / (m.sum(1) + m.sum(0) - m.diag())

  def get_mean_iou(self):
    """Returns the mean IoU."""
    return self.get_row_iou().mean()

  def __str__(self):
    acc_global = self.get_accuracy()
    acc = self.get_row_accuracy()
    recall = self.get_recall()
    iou = self.get_row_iou()
    f1_score = self.get_f1_score()
    return (
      f"global correct: {acc_global.item() * 100:.2f}\n"
      f"average row correct: {['{:.2f}'.format(i) for i in (acc * 100).tolist()]}\n"
      f"Recall: {['{:.2f}'.format(i) for i in (recall * 100).tolist()]}\n"
      f"IoU: {['{:.2f}'.format(i) for i in (iou * 100).tolist()]}\n"
      f"F1 Score: {['{:.2f}'.format(i) for i in (f1_score * 100).tolist()]}\n"
      f"mean IoU: {iou.mean().item() * 100:.2f}")
      
def centers_to_boxes(centers, width, height):
    centers = torch.tensor(centers, dtype=torch.float32)
    half_w, half_h = width / 2, height / 2
    boxes = torch.cat([centers - torch.tensor([half_w, half_h]),
                       centers + torch.tensor([half_w, half_h])], dim=1)
    return boxes

def calculate_iou_torchvision(boxes1, boxes2):
    return box_iou(boxes1, boxes2)

def evaluate_detection_raw(iou_matrix, iou_threshold=0.5):
    # Evaluate detection based on IoU matrix
    true_positives = 0
    matched_gt = set()
    matched_pred = set()

    for gt_idx in range(iou_matrix.size(0)):
        for pred_idx in range(iou_matrix.size(1)):
            if iou_matrix[gt_idx, pred_idx] >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
                true_positives += 1
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)

    precision = true_positives / iou_matrix.size(1) if iou_matrix.size(1) > 0 else 0
    recall = true_positives / iou_matrix.size(0) if iou_matrix.size(0) > 0 else 0
    return precision, recall

def evaluate_detection_raw_multiple(iou_matrices, iou_threshold=0.5):
    all_precision = []
    all_recall = []

    for iou_matrix in iou_matrices:
        true_positives = 0
        matched_gt = set()
        matched_pred = set()

        for gt_idx in range(iou_matrix.size(0)):
            for pred_idx in range(iou_matrix.size(1)):
                if iou_matrix[gt_idx, pred_idx] >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
                    true_positives += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)

        precision = true_positives / iou_matrix.size(1) if iou_matrix.size(1) > 0 else 0
        recall = true_positives / iou_matrix.size(0) if iou_matrix.size(0) > 0 else 0

        all_precision.append(precision)
        all_recall.append(recall)

    # Calculate average precision and recall across all images
    avg_precision = sum(all_precision) / len(all_precision) if all_precision else 0
    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    return avg_precision, avg_recall

from itertools import product

def evaluate_detection_raw_multiple(iou_matrices, iou_threshold=0.5):
    all_precision = []
    all_recall = []

    for iou_matrix in iou_matrices:
        # Get the highest IoU value and corresponding prediction for each ground truth
        max_vals, best_preds = torch.max(iou_matrix, dim=1)
        valid = max_vals >= iou_threshold  # Filter by threshold

        # Indices of ground truths with valid predictions
        valid_gt_indices = torch.arange(iou_matrix.size(0))[valid]
        valid_pred_indices = best_preds[valid]

        # Create a tensor to track which predictions have been matched
        matched_preds = torch.zeros(iou_matrix.size(1), dtype=torch.bool)

        true_positives = 0
        for gt_idx, pred_idx in zip(valid_gt_indices, valid_pred_indices):
            if not matched_preds[pred_idx]:
                matched_preds[pred_idx] = True  # Mark this prediction as matched
                true_positives += 1

        total_predictions = iou_matrix.size(1)
        total_gts = iou_matrix.size(0)
        precision = true_positives / total_predictions if total_predictions > 0 else 0
        recall = true_positives / total_gts if total_gts > 0 else 0

        all_precision.append(precision)
        all_recall.append(recall)

    avg_precision = torch.mean(torch.tensor(all_precision))
    avg_recall = torch.mean(torch.tensor(all_recall))

    return avg_precision.item(), avg_recall.item()

def evaluate_detection(iou_matrix, scores, iou_threshold):
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_iou_matrix = iou_matrix[:, sorted_indices]
    matched_gt = set()
    tp = 0
    precisions = []
    recalls = []
    #print(sorted_iou_matrix.shape)
    for i in range(sorted_iou_matrix.shape[1]):
        if sorted_iou_matrix[:, i].max() >= iou_threshold and sorted_iou_matrix[:, i].argmax() not in matched_gt:
            tp += 1
            matched_gt.add(sorted_iou_matrix[:, i].argmax())
        precision = tp / (i + 1)
        recall = tp / sorted_iou_matrix.shape[0]
        precisions.append(precision)
        recalls.append(recall)
    return torch.tensor(recalls), torch.tensor(precisions)

def calculate_ap(recalls, precisions):
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(precisions.size(0) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap

def calculate_mAP_per_image(boxes_gt, boxes_pred, scores, iou_thresholds=[0.5]):
    aps = []
    for iou_thresh in iou_thresholds:
        iou_matrix = box_iou(boxes_gt, boxes_pred)
        recalls, precisions = evaluate_detection(iou_matrix, scores, iou_thresh)
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
    return torch.mean(torch.tensor(aps))

def calculate_mAP_multiple_images(list_boxes_gt, list_boxes_pred, list_scores, iou_thresholds=[0.5]):
    # Expect lists of tensors for multiple images
    all_aps = []
    for boxes_gt, boxes_pred, scores in zip(list_boxes_gt, list_boxes_pred, list_scores):
        ap_per_image = calculate_mAP_per_image(boxes_gt, boxes_pred, scores, iou_thresholds)
        #print(ap_per_image)
        all_aps.append(ap_per_image)
    return torch.mean(torch.tensor(all_aps))

def f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    Compute the F_beta score given precision, recall, and beta.
    
    F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    Args:
        precision (float): Precision (P), in [0, 1].
        recall (float): Recall (R), in [0, 1].
        beta (float): Weight of recall relative to precision. Default is 1.0.
    
    Returns:
        float: The F_beta score.
    """
    if precision + recall == 0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall)