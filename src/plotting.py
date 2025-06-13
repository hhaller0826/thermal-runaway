
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

#Saving
def save_all_confusion_matrices(conf_mats, dir='./results'):
    _, axs = plt.subplots(3, 3, figsize=(18, 15))
    axs = axs.flatten()
    for ax, (name, cm) in zip(axs, conf_mats.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Healthy", "Runaway"],
                    yticklabels=["Healthy", "Runaway"], ax=ax)
        ax.set_title(f"Confusion: {name}")
    plt.tight_layout()
    plt.savefig('{}/all_confusion_matrices.png'.format(dir))
    plt.close()

def save_all_roc_curves(roc_data, dir='./results'):
    _, ax = plt.subplots(figsize=(10, 8))
    for name, (y_true, y_prob) in sorted(roc_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig('{}/all_roc_curves_sorted.png'.format(dir))
    plt.close()

def save_auc_comparison(train_aucs, test_aucs, dir='./results'):
    labels = list(train_aucs.keys())
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, [train_aucs[l] for l in labels], width, label="Train AUC")
    plt.bar(x + width/2, [test_aucs[l] for l in labels], width, label="Test AUC")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("AUC")
    plt.title("Train vs Test AUC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}/train_vs_test_auc.png'.format(dir))
    plt.close()

# Plotting
def plot_confusion_matrices(confusion_matrices):
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    axs = axs.flatten()
    for ax, (name, cm) in zip(axs, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Healthy", "Runaway"],
                    yticklabels=["Healthy", "Runaway"], ax=ax)
        ax.set_title(f"Confusion: {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc_curves(roc_curves):
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_roc = sorted(roc_curves.items(), key=lambda x: auc(*roc_curve(x[1][0], x[1][1])[:2]), reverse=True)
    for name, (y_true, y_prob) in sorted_roc:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_title("ROC Curves (Sorted by AUC)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.show()

def auc_comparison(train_auc, test_auc):
    labels = list(train_auc.keys())
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, [train_auc[k] for k in labels], width, label="Train AUC")
    plt.bar(x + width/2, [test_auc[k] for k in labels], width, label="Test AUC")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("AUC")
    plt.title("Train vs Test AUC Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()