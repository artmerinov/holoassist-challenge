import torch
from typing import List, Tuple


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