"""
utils/eval_utils.py
-------------------
Shared evaluation, plotting, interpretability, and reporting utilities.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2

Changes from original for Keras 3 compatibility:
    - All tensorflow.keras.backend (K.*) calls replaced with tf.* equivalents
    - GradientTape pattern updated for Keras 3 eager execution
    - Imports updated to use keras directly
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras

FAULT_LABELS = ['Normal', 'Fault-1', 'Fault-2', 'Fault-3',
                'Fault-4', 'Fault-5', 'Fault-6']

TRAIN_COLOR = '#2563EB'
VAL_COLOR   = '#F97316'
GRAD_CMAP   = 'RdYlGn_r'


# ─────────────────────────────────────────────────────────────────────
# 1. Model Evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test_onehot, y_test_int,
                   model_name: str = 'Model',
                   save_dir: str = 'outputs',
                   is_branch_input: bool = False):
    """
    Full evaluation: accuracy, confusion matrix, classification report.

    Args:
        model           : Trained Keras model.
        X_test          : Test features. For multi-branch CNN pass the
                          tf.data.Dataset or list of arrays.
        y_test_onehot   : One-hot labels (N, 7).
        y_test_int      : Integer labels (N,).
        model_name      : Used in titles and filenames.
        save_dir        : Output directory.
        is_branch_input : True for multi-branch CNN (list input).
    Returns:
        test_acc (float), y_pred_int (np.ndarray)
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')

    test_loss, test_acc = model.evaluate(X_test, verbose=0)
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*55}")

    y_pred     = model.predict(X_test, verbose=0)
    y_pred_int = np.argmax(y_pred, axis=1)

    report_str = classification_report(y_test_int, y_pred_int,
                                       target_names=FAULT_LABELS)
    print("\nPer-Class Classification Report:")
    print(report_str)

    report_dict = classification_report(y_test_int, y_pred_int,
                                        target_names=FAULT_LABELS,
                                        output_dict=True)
    pd.DataFrame(report_dict).T.to_csv(
        os.path.join(save_dir, f"{safe_name}_classification_report.csv")
    )

    cm_arr = confusion_matrix(y_test_int, y_pred_int)
    _plot_confusion_matrix(cm_arr, model_name=model_name, save_dir=save_dir)
    np.savetxt(os.path.join(save_dir, f"{safe_name}_confusion_matrix.csv"),
               cm_arr, delimiter=',', fmt='%d')

    return test_acc, y_pred_int


def _plot_confusion_matrix(cm_arr, model_name: str, save_dir: str):
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS,
                linewidths=0.5, annot_kws={'size': 10})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(save_dir, f"{safe_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# 2. Training History Curves
# ─────────────────────────────────────────────────────────────────────

def plot_training_history(history, model_name: str = 'Model',
                          save_dir: str = 'outputs'):
    """Plot and save training/validation accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} - Training History", fontsize=14, fontweight='bold')

    axes[0].plot(history.history['accuracy'],
                 label='Train', color=TRAIN_COLOR, linewidth=2)
    axes[0].plot(history.history['val_accuracy'],
                 label='Validation', color=VAL_COLOR, linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],
                 label='Train', color=TRAIN_COLOR, linewidth=2)
    axes[1].plot(history.history['val_loss'],
                 label='Validation', color=VAL_COLOR, linewidth=2, linestyle='--')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{safe_name}_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# 3. Grad-CAM for 1D CNN
# ─────────────────────────────────────────────────────────────────────

def gradcam_1d(model, sample_input, class_idx,
               last_conv_layer_name: str = None):
    """
    Compute Grad-CAM heatmap for a 1D CNN model.
    Updated for Keras 3: uses tf.GradientTape with explicit input watching.

    Args:
        model                : Trained Keras CNN model.
        sample_input         : List of 3 arrays (1,1000,1) for multi-branch,
                               or single array (1,1000,3) for hybrid.
        class_idx            : Target class index (0-6).
        last_conv_layer_name : Name of last Conv1D layer (auto-detected if None).

    Returns:
        heatmap: np.ndarray shape (T,) normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv1D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv1D layer found in model.")

    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    if isinstance(sample_input, list):
        inp = [tf.cast(tf.constant(s), tf.float32) for s in sample_input]
    else:
        inp = tf.cast(tf.constant(sample_input), tf.float32)

    with tf.GradientTape() as tape:
        if isinstance(inp, list):
            for i in inp:
                tape.watch(i)
        else:
            tape.watch(inp)
        conv_outputs, predictions = grad_model(inp, training=False)
        loss = predictions[:, class_idx]

    grads       = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out    = conv_outputs[0]
    heatmap     = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.nn.relu(heatmap).numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def plot_gradcam(model, sample, y_true_label, y_pred_label,
                 branch: int = 0,
                 last_conv_layer_name: str = None,
                 model_name: str = 'CNN',
                 save_dir: str = 'outputs',
                 is_branch_input: bool = True):
    """Overlay Grad-CAM heatmap on the raw vibration signal."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    axis_name = ['X', 'Y', 'Z'][branch]

    if is_branch_input:
        sample_input = [sample[i].reshape(1, 1000, 1) for i in range(3)]
        raw_signal   = sample[branch]
    else:
        sample_input = sample.transpose(1, 0).reshape(1, 1000, 3)
        raw_signal   = sample[branch]

    try:
        heatmap = gradcam_1d(model, sample_input,
                             np.argmax(model.predict(sample_input, verbose=0)),
                             last_conv_layer_name)
    except Exception as e:
        print(f"  Grad-CAM failed: {e}")
        return

    heatmap_up = np.interp(
        np.linspace(0, len(heatmap) - 1, 1000),
        np.arange(len(heatmap)), heatmap
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f'{model_name} - Grad-CAM | {axis_name}-Axis\n'
        f'True: {y_true_label}  |  Predicted: {y_pred_label}',
        fontsize=12, fontweight='bold'
    )
    t = np.arange(1000)
    axes[0].plot(t, raw_signal, color=TRAIN_COLOR, linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title('Raw Vibration Signal', fontsize=10)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t, raw_signal, color='black', linewidth=0.6, alpha=0.6, zorder=2)
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].set_xlabel('Time Step', fontsize=10)
    axes[1].set_title('Grad-CAM Overlay  (Red = High Model Attention)', fontsize=10)

    colormap = plt.get_cmap(GRAD_CMAP)
    for i in range(999):
        axes[1].axvspan(t[i], t[i+1],
                        color=colormap(heatmap_up[i]), alpha=0.4, linewidth=0)

    sm = plt.cm.ScalarMappable(cmap=GRAD_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label='Grad-CAM Intensity', pad=0.01)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{safe_name}_gradcam_{axis_name}axis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved Grad-CAM: {path}")
    plt.show()
    plt.close()


def run_gradcam_suite(model, X_test, y_test_int,
                      model_name: str = 'CNN',
                      save_dir: str = 'outputs',
                      is_branch_input: bool = True,
                      n_classes: int = 7):
    """Run Grad-CAM on one sample per fault class."""
    print(f"\n  Running Grad-CAM for {n_classes} fault classes...")
    for class_idx in range(n_classes):
        indices = np.where(y_test_int == class_idx)[0]
        if len(indices) == 0:
            print(f"  Class {class_idx}: no test samples, skipping.")
            continue
        sample   = X_test[indices[0]]
        y_true   = FAULT_LABELS[class_idx]
        inp      = ([sample[i].reshape(1, 1000, 1) for i in range(3)]
                    if is_branch_input
                    else sample.transpose(1, 0).reshape(1, 1000, 3))
        pred_idx = np.argmax(model.predict(inp, verbose=0), axis=1)[0]
        print(f"  Class {class_idx} ({y_true}): predicted as {FAULT_LABELS[pred_idx]}")
        plot_gradcam(model=model, sample=sample,
                     y_true_label=y_true, y_pred_label=FAULT_LABELS[pred_idx],
                     branch=0, model_name=f"{model_name}_class{class_idx}",
                     save_dir=save_dir, is_branch_input=is_branch_input)


# ─────────────────────────────────────────────────────────────────────
# 4. Attention Weight Visualization (RNN + Attention models)
# ─────────────────────────────────────────────────────────────────────

def extract_attention_weights(model, sample_input,
                              attention_layer_name: str = None):
    """
    Extract learned attention weights from AttentionLayer.
    Uses tf ops directly (no keras.backend) for Keras 3 compatibility.
    """
    if attention_layer_name is None:
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                attention_layer_name = layer.name
                break
        if attention_layer_name is None:
            print("  No AttentionLayer found.")
            return None
    try:
        attn_layer = model.get_layer(attention_layer_name)
        rnn_model  = keras.models.Model(
            inputs=model.inputs, outputs=attn_layer.input
        )
        hidden = rnn_model.predict(sample_input, verbose=0)  # (1, 1000, units)
        W = attn_layer.W.numpy()
        b = attn_layer.b.numpy()
        e = np.tanh(hidden[0] @ W + b)       # (1000, 1)
        e_exp   = np.exp(e - e.max())
        weights = (e_exp / e_exp.sum()).squeeze()
        return weights
    except Exception as ex:
        print(f"  Attention extraction failed: {ex}")
        return None


def plot_attention_weights(model, sample, y_true_label, y_pred_label,
                           model_name: str = 'RNN+Attention',
                           save_dir: str = 'outputs',
                           attention_layer_name: str = None):
    """Plot attention weights over 1000 timesteps alongside the raw signal."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name  = model_name.replace(' ', '_').replace('+', '_')
    sample_rnn = sample.transpose(1, 0).reshape(1, 1000, 3).astype(np.float32)
    weights    = extract_attention_weights(model, sample_rnn, attention_layer_name)

    if weights is None:
        print(f"  Skipping attention plot for {model_name}.")
        return

    t   = np.arange(1000)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f'{model_name} - Attention Weights\n'
        f'True: {y_true_label}  |  Predicted: {y_pred_label}',
        fontsize=12, fontweight='bold'
    )
    for i, (color, label) in enumerate(
        zip(['#2563EB', '#16A34A', '#DC2626'], ['X', 'Y', 'Z'])
    ):
        axes[0].plot(t, sample[i], color=color, linewidth=0.7, alpha=0.8, label=f'{label}-axis')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].fill_between(t, weights, alpha=0.6, color='#DC2626', label='Attention')
    axes[1].plot(t, weights, color='#7F1D1D', linewidth=0.8)
    threshold = np.percentile(weights, 95)
    axes[1].fill_between(t, weights, where=weights >= threshold,
                         alpha=0.9, color='#991B1B', label='Top 5%')
    axes[1].set_ylabel('Attention Weight', fontsize=10)
    axes[1].set_xlabel('Time Step', fontsize=10)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{safe_name}_attention_weights.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved attention plot: {path}")
    plt.show()
    plt.close()


def run_attention_suite(model, X_test, y_test_int,
                        model_name: str = 'RNN+Attention',
                        save_dir: str = 'outputs',
                        n_samples: int = 3):
    """Plot attention weights for n_samples from different fault classes."""
    print(f"\n  Visualizing attention for {n_samples} samples...")
    shown = 0
    for class_idx in range(7):
        if shown >= n_samples:
            break
        indices = np.where(y_test_int == class_idx)[0]
        if len(indices) == 0:
            continue
        sample = X_test[indices[0]]
        inp    = sample.transpose(1, 0).reshape(1, 1000, 3).astype(np.float32)
        pred   = np.argmax(model.predict(inp, verbose=0), axis=1)[0]
        plot_attention_weights(
            model=model, sample=sample,
            y_true_label=FAULT_LABELS[class_idx],
            y_pred_label=FAULT_LABELS[pred],
            model_name=f"{model_name}_sample_class{class_idx}",
            save_dir=save_dir,
        )
        shown += 1


# ─────────────────────────────────────────────────────────────────────
# 5. Multi-Model Comparison
# ─────────────────────────────────────────────────────────────────────

def compare_models_boxplot(results_dict: dict,
                           title: str = 'Model Accuracy Comparison',
                           save_dir: str = 'outputs',
                           filename: str = 'model_comparison_boxplot.png'):
    """Boxplot comparing multiple models across N runs."""
    os.makedirs(save_dir, exist_ok=True)
    df     = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_dict.items()]))
    melted = df.melt(var_name='Model', value_name='Accuracy')

    plt.figure(figsize=(max(10, len(results_dict) * 1.4), 6))
    sns.boxplot(x='Model', y='Accuracy', data=melted,
                palette='Set2', width=0.5, linewidth=1.5)
    plt.xticks(rotation=40, ha='right', fontsize=9)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.ylim(max(0, melted['Accuracy'].min() - 0.05), 1.02)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved boxplot: {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# 6. Summary Table
# ─────────────────────────────────────────────────────────────────────

def print_summary_table(results: list, title: str = 'MODEL RESULTS SUMMARY'):
    df = pd.DataFrame(results).sort_values('Best_Acc', ascending=False).reset_index(drop=True)
    for col in ['Best_Acc', 'Mean_Acc']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    if 'Std_Acc' in df.columns:
        df['Std_Acc'] = df['Std_Acc'].apply(lambda x: f"+/-{x*100:.2f}%")
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}\n")
    return df

def save_summary_csv(results: list, filename: str, save_dir: str = 'outputs'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"  Saved summary CSV: {path}")


# ─────────────────────────────────────────────────────────────────────
# 7. Model Persistence
# ─────────────────────────────────────────────────────────────────────

def save_model(model, model_name: str, save_dir: str = 'outputs'):
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    path = os.path.join(save_dir, f"{safe_name}.keras")
    model.save(path)
    print(f"  Model saved: {path}")
    return path
