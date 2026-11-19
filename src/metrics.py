import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, top_k_accuracy_score

def calculate_metrics(y_true, y_pred, y_probs=None, class_indices=None):
    """
    Computes Top-1, Top-5, Macro F1, Per-class Precision, Recall, F1, and Confusion Matrix.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    top1 = accuracy_score(y_true, y_pred)
    top5 = None
    if y_probs is not None:
        try:
            if y_probs.shape[1] >= 5:
                top5 = top_k_accuracy_score(y_true, y_probs, k=min(5, y_probs.shape[1]), labels=np.arange(y_probs.shape[1]))
            else:
                top5 = top1
        except Exception:
            top5 = None
            
    if class_indices is None:
        class_indices = np.unique(np.concatenate([y_true, y_pred]))
        
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, zero_division=0
    )
    
    macro_f1 = np.mean(f1)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    
    cm = confusion_matrix(y_true, y_pred, labels=class_indices)
    
    per_class = {}
    for i, idx in enumerate(class_indices):
        per_class[int(idx)] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
        
    return {
        'top1_accuracy': float(top1),
        'top5_accuracy': float(top5) if top5 is not None else float(top1),
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'per_class': per_class,
        'confusion_matrix': cm
    }

def print_metrics_table(metrics_dict, class_names=None):
    print("=" * 65)
    print(f"Top-1 Accuracy: {metrics_dict['top1_accuracy']*100:.2f}%")
    if 'top5_accuracy' in metrics_dict and metrics_dict['top5_accuracy'] is not None:
        print(f"Top-5 Accuracy: {metrics_dict['top5_accuracy']*100:.2f}%")
    print(f"Macro F1-Score: {metrics_dict['macro_f1']*100:.2f}%")
    print(f"Macro Precision: {metrics_dict['macro_precision']*100:.2f}%")
    print(f"Macro Recall:   {metrics_dict['macro_recall']*100:.2f}%")
    print("-" * 65)
    print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 65)
    for c_id, stats in metrics_dict['per_class'].items():
        name = class_names[c_id] if class_names and c_id < len(class_names) else f"Class {c_id}"
        print(f"{name:<25} {stats['precision']*100:>8.2f}% {stats['recall']*100:>8.2f}% {stats['f1']*100:>8.2f}% {stats['support']:>8}")
    print("=" * 65)
