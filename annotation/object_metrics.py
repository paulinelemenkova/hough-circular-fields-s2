import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(pred, ref):
    p = pred.reshape(len(pred), -1).astype(np.float32)
    r = ref.reshape(len(ref), -1).astype(np.float32)
    inter = p @ r.T
    union = p.sum(1)[:, None] + r.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def object_scores(pred, ref, thr=0.5):
    if len(pred) == 0 or len(ref) == 0:
        tp = 0
    else:
        iou = iou_matrix(pred, ref)
        rows, cols = linear_sum_assignment(-iou)
        tp = int((iou[rows, cols] >= thr).sum())
    fp, fn = len(pred) - tp, len(ref) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "F1": f1}
