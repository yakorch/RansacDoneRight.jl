import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def plot_roc_curves(classifiers: list[np.ndarray], labels: list[str], ground_truth: np.ndarray, safe_path: str = None):
    """
    - `ground_truth` is common for all classifiers
    - `labels` and `classfiers` should be in the same order
    - The higher value each classifier returned, the more likely `H_0` is to be rejected.
    - `ground_truth` must consist of zeros and ones only. `1` means `H_0` is true, while `0` means `H_1` is true.
    """
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'black']
    plt.figure()

    for (cdf_values, label, color) in zip(classifiers, labels, colors):
        fpr, tpr, thresholds = roc_curve(ground_truth, cdf_values, pos_label=0)
        roc_auc1 = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (area = %0.6f)' % roc_auc1)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC Consensus Set Labelling Plot')
    plt.legend(loc="lower right")

    if safe_path is not None:
        plt.savefig(safe_path)
    plt.show()


def test():
    labels = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0])  # 1 for H0, 0 for H1
    V_1 = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.05, 0.9, 0.6, 0.3, 0.7])
    V_2 = np.array([0.05, 0.3, 0.25, 0.9, 0.15, 0.65, 0.35, 0.5, 0.45, 0.85])  # Predictions from model 2
    plot_roc_curves([V_1, V_2], ['Model 1', 'Model 2'], labels)
