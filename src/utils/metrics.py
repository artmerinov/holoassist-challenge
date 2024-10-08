import torch
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt


def calc_accuracy(preds: torch.Tensor, 
                  labels: torch.Tensor, 
                  topk: Tuple[int] = (1,),
    ) -> List[float]:
    """
    Computes precision@k.

    preds: 
        shape [batch_size, num_classes]
    labels: 
        shape [batch_size]

    Example:
    --------
    preds = torch.tensor(
        [[0.80, 0.15, 0.01, 0.00, 0.04],
        [0.35, 0.40, 0.20, 0.05, 0.00]]
    )
    labels = torch.tensor([1, 4])
    topk_values = (1, 2, 3, 4, 5)
    result = [0.0, 0.5, 0.5, 0.5, 1.0]
    """
    with torch.no_grad():
        
        max_k = max(topk)
        batch_size = len(labels)

        # Return the k largest elements and its corresponding indeces
        _, largest_ids = torch.topk(preds, k=max_k, dim=1, largest=True, sorted=True)
        is_correct = largest_ids.t() == labels.expand_as(largest_ids.t())

        accuracies = []
        for k in topk:
            batch_num_correct_at_k = is_correct[:k].sum()
            batch_accuracy_at_k = (batch_num_correct_at_k / batch_size).item()
            accuracies.append(batch_accuracy_at_k)

    return accuracies


def find_threshold(y_true: torch.Tensor, 
                   y_score: torch.Tensor) -> float:
    """
    Finds threshold that maximises F1 score.

    Args:
        y_true: Ground truth labels.
        y_score: Probabilities of positive class.
    """
    thr_list = torch.unique(y_score)
    f1score_list = torch.empty(len(thr_list), dtype=torch.float32)

    for i, thr in enumerate(thr_list):

        tp, tn, fp, fn, precision, recall, f1score = calc_thr_metrics(
            thr=thr,
            y_true=y_true,
            y_score=y_score,
        )
        f1score_list[i] = f1score

    # Choose threshold that maximise F1
    index = torch.argmax(f1score_list)
    thr = thr_list[index]

    return thr


def calc_thr_metrics(
        thr: float,
        y_true: torch.Tensor, 
        y_score: torch.Tensor,
    ) -> Tuple[float, ...]:
    """
    Calculate basic binary classification metrics 
    based on threshold.

    Args:
        thr: Threshold for label prediction.
        y_true: Ground truth labels.
        y_score: Probabilities of positive class.
    """
    y_pred = (y_score >= thr).int()

    tp = torch.sum((y_true == 1) & (y_pred == 1))
    tn = torch.sum((y_true == 0) & (y_pred == 0))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    fn = torch.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1score = 2 * precision * recall / (precision + recall + 1e-6)

    return tp, tn, fp, fn, precision, recall, f1score
    

def save_clf_visualisation(
        y_true: torch.Tensor, 
        y_score: torch.Tensor, 
        fname: str,
    ) -> None:
    """
    Save visualisation of binary classification.

    Args:
        y_true: Ground truth labels.
        y_score: Probabilities of positive class.
        fname: Path to save image.
    """
    thr_list = torch.unique(y_score)

    f1score_list = torch.empty(len(thr_list), dtype=torch.float32)
    precision_list = torch.empty(len(thr_list), dtype=torch.float32)
    recall_list = torch.empty(len(thr_list), dtype=torch.float32)

    for i, thr in enumerate(thr_list):

        tp, tn, fp, fn, precision, recall, f1score = calc_thr_metrics(
            thr=thr,
            y_true=y_true,
            y_score=y_score,
        )
        
        f1score_list[i] = f1score
        precision_list[i] = precision
        recall_list[i] = recall

    index = torch.argmax(f1score_list)

    # PLOT

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20,5))

    ax1.plot(recall_list, precision_list, linestyle='--', marker='.', color='k', lw=0.5, markersize=2)
    ax1.plot(recall_list[index], precision_list[index], marker='x', color='red', markersize=10)
    ax1.set_title("Precision (recall)", fontweight='bold')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')

    ax2.plot(thr_list, precision_list, linestyle='--', marker='.', color='teal', lw=0.5, markersize=2, label="Precision")
    ax2.plot(thr_list, recall_list, linestyle='--', marker='.', color='indigo', lw=0.5, markersize=2, label="Recall")
    ax2.legend()
    ax2.plot(thr_list[index], precision_list[index], marker='x', color='red', markersize=10)
    ax2.plot(thr_list[index], recall_list[index], marker='x', color='red', markersize=10)
    ax2.set_title("Precision & Recall (threshold)", fontweight='bold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')

    ax3.plot(thr_list, f1score_list, linestyle='--', marker='.', color='k', lw=0.5, markersize=2, label="F1score")
    ax3.plot(thr_list[index], f1score_list[index], marker='x', color='red', markersize=10)
    ax3.set_title("F1score (threshold)", fontweight='bold')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1score')

    title = f"thr={thr_list[index]:.3f}", \
            f"Precision={precision_list[index]:.3f}", \
            f"Recall={recall_list[index]:.3f}", \
            f"F1score={f1score_list[index]:.3f}", \
            
    fig.suptitle(title, fontsize=16, fontweight='bold');
    plt.close(fig)

    fig.savefig(fname, dpi=200)
