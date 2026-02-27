"""
utils/eval_utils.py
-------------------
Shared evaluation, plotting, and reporting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

FAULT_LABELS = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3', 'Fault-4', 'Fault-5', 'Fault-6']


def save_model(model, model_name: str, save_dir: str = 'outputs'):
    """Save trained Keras model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.keras")
    model.save(path)
    print(f"Model saved: {path}")
    return path


def evaluate_model(model, X_test, y_test_onehot, y_test_int,
                   model_name: str = "Model",
                   save_dir: str = None,
                   is_branch_input: bool = False):
    """
    Full evaluation pipeline:
      - Test accuracy / loss
      - Classification report (precision, recall, F1 per class)
      - Confusion matrix heatmap

    Args:
        model:           Trained Keras model.
        X_test:          Test features (list of arrays if is_branch_input=True).
        y_test_onehot:   One-hot encoded true labels.
        y_test_int:      Integer true labels.
        model_name:      Used in plot titles and saved filenames.
        save_dir:        Directory to save plots/reports. None = don't save.
        is_branch_input: True for multi-branch CNN (list input).
    """
    test_input = X_test if is_branch_input else X_test

    # ── Accuracy
    test_loss, test_acc = model.evaluate(test_input, y_test_onehot, verbose=0)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*50}\n")

    # ── Predictions
    y_pred = model.predict(test_input, verbose=0)
    y_pred_int = np.argmax(y_pred, axis=1)

    # ── Classification Report
    print("Per-Class Classification Report:")
    print(classification_report(y_test_int, y_pred_int, target_names=FAULT_LABELS))

    # ── Confusion Matrix
    cm = confusion_matrix(y_test_int, y_pred_int)
    _plot_confusion_matrix(cm, model_name, save_dir)

    # ── Save outputs to CSV
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = model_name.replace(' ', '_')

        # Classification report as CSV
        rep = classification_report(y_test_int, y_pred_int,
                                    target_names=FAULT_LABELS, output_dict=True)
        import pandas as pd
        pd.DataFrame(rep).T.to_csv(
            os.path.join(save_dir, f"{safe_name}_classification_report.csv")
        )

        # Confusion matrix as CSV
        np.savetxt(
            os.path.join(save_dir, f"{safe_name}_confusion_matrix.csv"),
            cm, delimiter=',', fmt='%d'
        )
        print(f"Saved reports: {save_dir}/{safe_name}_*.csv")

    return test_acc, y_pred_int


def plot_training_history(history, model_name: str = "Model", save_dir: str = None):
    """Plot accuracy and loss curves — blue/orange colors, linewidth=2 (matches original style)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy',
                 color='blue',   linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy',
                 color='orange', linewidth=2)
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss',
                 color='blue',   linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss',
                 color='orange', linewidth=2)
    axes[1].set_title('Training vs Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.show()


def _plot_confusion_matrix(cm, model_name: str, save_dir: str = None, cmap: str = 'Blues'):
    """Internal: plot and optionally save confusion matrix."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS,
        linewidths=0.5
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{model_name} — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.show()


def compare_models_boxplot(results_dict: dict, save_dir: str = None):
    """
    Boxplot comparing multiple models across N runs.

    Args:
        results_dict: {model_name: [acc_run1, acc_run2, ...], ...}
    """
    import pandas as pd

    df = pd.DataFrame(results_dict)
    melted = df.melt(var_name='Model', value_name='Accuracy')

    plt.figure(figsize=(14, 6))
    sns.boxplot(x='Model', y='Accuracy', data=melted, palette='Set2')
    plt.xticks(rotation=40, ha='right')
    plt.title('Model Accuracy Comparison (5 Runs Each)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'model_comparison_boxplot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    plt.show()
