"""
utils/eval_utils.py
-------------------
Shared evaluation, plotting, interpretability, and reporting utilities.
Used across ALL scripts in this project.

Contains:
    - evaluate_model()          : accuracy, confusion matrix, classification report
    - plot_training_history()   : training/validation curves
    - gradcam_1d()              : Grad-CAM for 1D CNN models
    - plot_gradcam()            : visualize Grad-CAM heatmap over raw signal
    - extract_attention_weights(): pull learned attention weights from AttentionLayer
    - plot_attention_weights()  : plot attention over 1000 timesteps
    - compare_models_boxplot()  : multi-model accuracy boxplot
    - print_summary_table()     : end-of-script results table
    - save_model()              : save Keras model to disk

Design principle:
    Every function saves to disk AND displays inline.
    Every model calls these immediately after training — not in a second pass.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.keras.backend as K

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

FAULT_LABELS = [
    'Normal',
    'Fault-1',
    'Fault-2',
    'Fault-3',
    'Fault-4',
    'Fault-5',
    'Fault-6',
]

# Colour palette — consistent across all plots
TRAIN_COLOR = '#2563EB'   # blue
VAL_COLOR   = '#F97316'   # orange
GRAD_CMAP   = 'RdYlGn_r' # red=high attention, green=low


# ─────────────────────────────────────────────────────────────────────
# 1. Model Evaluation  (accuracy + confusion matrix + report)
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test_onehot, y_test_int,
                   model_name: str = 'Model',
                   save_dir: str = 'outputs',
                   is_branch_input: bool = False):
    """
    Full inline evaluation pipeline — called immediately after training.

    Steps performed here:
        1. Compute test accuracy and loss
        2. Print per-class classification report
        3. Plot and save confusion matrix
        4. Save classification report CSV

    Args:
        model           : Trained Keras model.
        X_test          : Test features.
                          List of arrays if is_branch_input=True (multi-branch CNN).
        y_test_onehot   : One-hot encoded true labels, shape (N, 7).
        y_test_int      : Integer class labels, shape (N,).
        model_name      : Used in titles and filenames.
        save_dir        : Directory for all saved outputs.
        is_branch_input : True for multi-branch CNN models (list input).

    Returns:
        test_acc   (float)
        y_pred_int (np.ndarray) — predicted integer class labels
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')

    # ── 1. Accuracy & Loss
    test_input = X_test  # works for both list and array inputs
    test_loss, test_acc = model.evaluate(test_input, y_test_onehot, verbose=0)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*55}")

    # ── 2. Predictions
    y_pred       = model.predict(test_input, verbose=0)
    y_pred_int   = np.argmax(y_pred, axis=1)

    # ── 3. Classification report (console + CSV)
    report_str  = classification_report(
        y_test_int, y_pred_int, target_names=FAULT_LABELS
    )
    print("\nPer-Class Classification Report:")
    print(report_str)

    report_dict = classification_report(
        y_test_int, y_pred_int,
        target_names=FAULT_LABELS, output_dict=True
    )
    pd.DataFrame(report_dict).T.to_csv(
        os.path.join(save_dir, f"{safe_name}_classification_report.csv")
    )

    # ── 4. Confusion matrix
    cm_arr = confusion_matrix(y_test_int, y_pred_int)
    _plot_confusion_matrix(cm_arr, model_name=model_name, save_dir=save_dir)

    # Save confusion matrix CSV
    np.savetxt(
        os.path.join(save_dir, f"{safe_name}_confusion_matrix.csv"),
        cm_arr, delimiter=',', fmt='%d'
    )

    return test_acc, y_pred_int


def _plot_confusion_matrix(cm_arr, model_name: str, save_dir: str):
    """Plot and save confusion matrix heatmap."""
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_arr, annot=True, fmt='d', cmap='Blues',
        xticklabels=FAULT_LABELS, yticklabels=FAULT_LABELS,
        linewidths=0.5, annot_kws={'size': 10}
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'{model_name} — Confusion Matrix', fontsize=13, fontweight='bold')
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

def plot_training_history(history, model_name: str = 'Model', save_dir: str = 'outputs'):
    """
    Plot and save training/validation accuracy and loss curves.
    Called immediately after model.fit() — before any other evaluation.

    Args:
        history    : Keras History object returned by model.fit().
        model_name : Used in title and filename.
        save_dir   : Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],
                 label='Train', color=TRAIN_COLOR, linewidth=2)
    axes[0].plot(history.history['val_accuracy'],
                 label='Validation', color=VAL_COLOR, linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
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
# 3. Grad-CAM for 1D CNN  (interpretability)
# ─────────────────────────────────────────────────────────────────────

def gradcam_1d(model, sample_input, class_idx,
               last_conv_layer_name: str = None):
    """
    Compute Grad-CAM heatmap for a 1D CNN model.

    Grad-CAM works by:
        1. Running a forward pass and recording the activations of the
           last convolutional layer.
        2. Computing the gradient of the target class score with respect
           to those activations.
        3. Pooling the gradients over the time dimension (global average).
        4. Weighting each filter's activation map by its pooled gradient.
        5. ReLU-ing the result → only positive contributions highlighted.

    For vibration fault detection: the heatmap shows WHICH TIME WINDOWS
    in the raw signal drove the model toward predicting a specific fault.
    This is what maintenance engineers need to trust the model.

    Args:
        model               : Trained Keras CNN model.
        sample_input        : Single sample input.
                              List of 3 arrays shaped (1, 1000, 1) for
                              multi-branch CNN; array (1, 1000, 3) for hybrid.
        class_idx           : Target class index (0–6) to explain.
        last_conv_layer_name: Name of the last Conv1D layer to hook into.
                              If None, auto-detected by scanning model layers.

    Returns:
        heatmap: np.ndarray of shape (T,) — normalized to [0, 1].
                 Length T matches the temporal output of the target layer.
    """
    # Auto-detect last Conv1D layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv1D):
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv1D layer found in model.")

    # Build a model that outputs: [last conv activations, final predictions]
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Prepare inputs
    if isinstance(sample_input, list):
        inp = [tf.cast(s, tf.float32) for s in sample_input]
    else:
        inp = tf.cast(sample_input, tf.float32)

    # Use persistent tape so we can compute gradient after the with-block
    with tf.GradientTape() as tape:
        if isinstance(inp, list):
            [tape.watch(i) for i in inp]
        else:
            tape.watch(inp)
        conv_outputs, predictions = grad_model(inp)
        loss = predictions[:, class_idx]

    # Gradient of class score w.r.t. last conv layer activations
    grads = tape.gradient(loss, conv_outputs)  # shape: (1, T, filters)

    # Pool gradients across time → importance weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # shape: (filters,)

    # Weight activations by pooled gradients
    conv_outputs = conv_outputs[0]                      # shape: (T, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (T, 1)
    heatmap = tf.squeeze(heatmap)                       # (T,)

    # ReLU: keep only features that push score UP for this class
    heatmap = tf.nn.relu(heatmap).numpy()

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def plot_gradcam(model, sample, y_true_label, y_pred_label,
                 branch: int = 0,
                 last_conv_layer_name: str = None,
                 model_name: str = 'CNN',
                 save_dir: str = 'outputs',
                 is_branch_input: bool = True):
    """
    Overlay Grad-CAM heatmap on the raw vibration signal.

    The output shows:
        - Raw signal (blue line)
        - Grad-CAM heatmap as a coloured background overlay
        - Red regions = time windows the model relied on most
        - Green regions = time windows the model largely ignored

    This is the visualization that makes the model explainable to
    a maintenance engineer — they can see WHERE in the signal the
    fault signature was detected.

    Args:
        model               : Trained Keras CNN model.
        sample              : Single sample, shape (3, 1000) — raw (not batched).
        y_true_label        : String label for true class.
        y_pred_label        : String label for predicted class.
        branch              : Which axis to visualize (0=X, 1=Y, 2=Z).
        last_conv_layer_name: Name of last Conv1D layer (auto-detected if None).
        model_name          : Used in title and filename.
        save_dir            : Directory to save figure.
        is_branch_input     : True for multi-branch CNN.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    axis_name = ['X', 'Y', 'Z'][branch]

    # Prepare batched input
    if is_branch_input:
        sample_input = [
            sample[i].reshape(1, 1000, 1) for i in range(3)
        ]
        raw_signal = sample[branch]
        pred_class = np.argmax(
            model.predict(sample_input, verbose=0), axis=1
        )[0]
    else:
        sample_input = sample.transpose(1, 0).reshape(1, 1000, 3)
        raw_signal   = sample[branch]
        pred_class   = np.argmax(
            model.predict(sample_input, verbose=0), axis=1
        )[0]

    # Compute Grad-CAM heatmap
    try:
        heatmap = gradcam_1d(
            model, sample_input, pred_class, last_conv_layer_name
        )
    except Exception as e:
        print(f"  Grad-CAM computation failed: {e}")
        return

    # Upsample heatmap to match signal length (1000)
    heatmap_upsampled = np.interp(
        np.linspace(0, len(heatmap) - 1, 1000),
        np.arange(len(heatmap)),
        heatmap
    )

    # ── Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f'{model_name} — Grad-CAM | {axis_name}-Axis\n'
        f'True: {y_true_label}  |  Predicted: {y_pred_label}',
        fontsize=12, fontweight='bold'
    )

    t = np.arange(1000)

    # Top: raw signal
    axes[0].plot(t, raw_signal, color=TRAIN_COLOR, linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title('Raw Vibration Signal', fontsize=10)
    axes[0].grid(True, alpha=0.2)

    # Bottom: signal + heatmap overlay
    axes[1].plot(t, raw_signal, color='black', linewidth=0.6, alpha=0.6, zorder=2)
    axes[1].set_ylabel('Amplitude', fontsize=10)
    axes[1].set_xlabel('Time Step', fontsize=10)
    axes[1].set_title('Grad-CAM Overlay  (Red = High Model Attention)', fontsize=10)

    # Coloured background overlay per time step
    colormap = plt.get_cmap(GRAD_CMAP)
    for i in range(999):
        axes[1].axvspan(
            t[i], t[i + 1],
            color=colormap(heatmap_upsampled[i]),
            alpha=0.4, linewidth=0
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=GRAD_CMAP,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
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
    """
    Run Grad-CAM on one representative sample from each fault class.
    Called inline immediately after evaluate_model() for CNN models.

    Produces n_classes × 1 Grad-CAM plots (one per fault type),
    always using the X-axis branch (branch=0) for consistency.

    Args:
        model           : Trained CNN model.
        X_test          : Test set, shape (N, 3, 1000).
        y_test_int      : Integer class labels, shape (N,).
        model_name      : Used in filenames and titles.
        save_dir        : Output directory.
        is_branch_input : True for multi-branch CNN.
        n_classes       : Number of fault classes (default 7).
    """
    print(f"\n  Running Grad-CAM for {n_classes} fault classes...")

    for class_idx in range(n_classes):
        # Find first sample of this class in test set
        indices = np.where(y_test_int == class_idx)[0]
        if len(indices) == 0:
            print(f"  Class {class_idx}: no test samples found, skipping.")
            continue

        sample  = X_test[indices[0]]   # shape: (3, 1000)
        y_true  = FAULT_LABELS[class_idx]

        # Get model prediction for this sample
        if is_branch_input:
            inp = [sample[i].reshape(1, 1000, 1) for i in range(3)]
        else:
            inp = sample.transpose(1, 0).reshape(1, 1000, 3)

        pred_idx   = np.argmax(model.predict(inp, verbose=0), axis=1)[0]
        y_pred     = FAULT_LABELS[pred_idx]

        print(f"  Class {class_idx} ({y_true}): predicted as {y_pred}")
        plot_gradcam(
            model        = model,
            sample       = sample,
            y_true_label = y_true,
            y_pred_label = y_pred,
            branch       = 0,
            model_name   = f"{model_name}_class{class_idx}",
            save_dir     = save_dir,
            is_branch_input = is_branch_input
        )


# ─────────────────────────────────────────────────────────────────────
# 4. Attention Weight Visualization  (for RNN + Attention models)
# ─────────────────────────────────────────────────────────────────────

def extract_attention_weights(model, sample_input,
                              attention_layer_name: str = None):
    """
    Extract the learned attention weights from an AttentionLayer.

    The AttentionLayer computes a softmax distribution over 1000 timesteps.
    Extracting and plotting these weights shows WHICH timesteps the model
    weighted most heavily when making its classification decision.

    If the model attends to fault-relevant windows → the architecture is
    working correctly. If it attends to noise → the model is not learning
    what we think it is. This is the scientific validation the original
    work skipped.

    Args:
        model                : Trained Keras model with AttentionLayer.
        sample_input         : Single sample, shape (1, 1000, 3).
        attention_layer_name : Name of the AttentionLayer.
                               Auto-detected if None.

    Returns:
        weights: np.ndarray shape (1000,) — attention weights summing to ~1.
                 None if no AttentionLayer found.
    """
    # Auto-detect AttentionLayer
    if attention_layer_name is None:
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                attention_layer_name = layer.name
                break
        if attention_layer_name is None:
            print("  No AttentionLayer found in model.")
            return None

    # Build sub-model that outputs attention weights
    # The AttentionLayer stores weights as the softmax output before summing
    try:
        # Hook into the layer that feeds INTO the AttentionLayer
        # to extract pre-attention hidden states
        attn_layer = model.get_layer(attention_layer_name)

        # Build a model that outputs the layer BEFORE attention
        # (the RNN hidden states, shape: (1, 1000, units))
        rnn_output_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = attn_layer.input   # RNN hidden states feeding attention
        )
        hidden_states = rnn_output_model.predict(sample_input, verbose=0)
        # hidden_states shape: (1, 1000, units)

        # Recompute attention weights using the layer's learned parameters
        W = attn_layer.W.numpy()   # (units, 1)
        b = attn_layer.b.numpy()   # (1000, 1)

        # e_t = tanh(h_t @ W + b)
        e = np.tanh(hidden_states[0] @ W + b)   # (1000, 1)
        # a_t = softmax(e_t) over time axis
        e_exp = np.exp(e - e.max())
        weights = (e_exp / e_exp.sum()).squeeze()  # (1000,)

        return weights

    except Exception as ex:
        print(f"  Attention weight extraction failed: {ex}")
        return None


def plot_attention_weights(model, sample, y_true_label, y_pred_label,
                           model_name: str = 'RNN+Attention',
                           save_dir: str = 'outputs',
                           attention_layer_name: str = None):
    """
    Plot attention weights over 1000 timesteps alongside the raw signal.

    Two-panel figure:
        Top    : All 3 raw vibration axes (X, Y, Z) overlaid.
        Bottom : Attention weight distribution over time.
                 Peaks = timesteps the model focused on.
                 This is the scientific validation of the AttentionLayer.

    Args:
        model                : Trained Keras model with AttentionLayer.
        sample               : Single sample, shape (3, 1000).
        y_true_label         : String label for true class.
        y_pred_label         : String label for predicted class.
        model_name           : Used in title and filename.
        save_dir             : Output directory.
        attention_layer_name : AttentionLayer name (auto-detected if None).
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')

    # RNN input format: (1, 1000, 3)
    sample_rnn = sample.transpose(1, 0).reshape(1, 1000, 3)
    sample_rnn = sample_rnn.astype(np.float32)

    weights = extract_attention_weights(
        model, sample_rnn, attention_layer_name
    )

    if weights is None:
        print(f"  Skipping attention plot for {model_name}.")
        return

    t = np.arange(1000)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        f'{model_name} — Attention Weights\n'
        f'True: {y_true_label}  |  Predicted: {y_pred_label}',
        fontsize=12, fontweight='bold'
    )

    # Top: raw signal (all 3 axes)
    axis_colors = ['#2563EB', '#16A34A', '#DC2626']
    for i, (color, label) in enumerate(zip(axis_colors, ['X', 'Y', 'Z'])):
        axes[0].plot(t, sample[i], color=color, linewidth=0.7,
                     alpha=0.8, label=f'{label}-axis')
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title('Raw Vibration Signal (3 Axes)', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Bottom: attention weights
    axes[1].fill_between(t, weights, alpha=0.6, color='#DC2626', label='Attention')
    axes[1].plot(t, weights, color='#7F1D1D', linewidth=0.8)

    # Mark top-5% attention peaks
    threshold = np.percentile(weights, 95)
    peak_mask = weights >= threshold
    axes[1].fill_between(t, weights, where=peak_mask,
                         alpha=0.9, color='#991B1B', label='Top 5% attention')

    axes[1].set_ylabel('Attention Weight', fontsize=10)
    axes[1].set_xlabel('Time Step (out of 1000)', fontsize=10)
    axes[1].set_title(
        'Attention Distribution — Peaks show where the model focused',
        fontsize=10
    )
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
    """
    Plot attention weights for n_samples representative test samples.
    Called inline after evaluate_model() for all attention-based models.

    Samples are chosen from different fault classes for variety.

    Args:
        model      : Trained Keras model with AttentionLayer.
        X_test     : Test set, shape (N, 3, 1000).
        y_test_int : Integer class labels.
        model_name : Used in filenames.
        save_dir   : Output directory.
        n_samples  : Number of sample visualizations to generate.
    """
    print(f"\n  Visualizing attention weights for {n_samples} samples...")

    # Pick one sample per class (up to n_samples)
    shown = 0
    for class_idx in range(7):
        if shown >= n_samples:
            break
        indices = np.where(y_test_int == class_idx)[0]
        if len(indices) == 0:
            continue

        sample = X_test[indices[0]]   # (3, 1000)
        inp    = sample.transpose(1, 0).reshape(1, 1000, 3).astype(np.float32)
        pred   = np.argmax(model.predict(inp, verbose=0), axis=1)[0]

        plot_attention_weights(
            model        = model,
            sample       = sample,
            y_true_label = FAULT_LABELS[class_idx],
            y_pred_label = FAULT_LABELS[pred],
            model_name   = f"{model_name}_sample_class{class_idx}",
            save_dir     = save_dir,
        )
        shown += 1


# ─────────────────────────────────────────────────────────────────────
# 5. Multi-Model Comparison
# ─────────────────────────────────────────────────────────────────────

def compare_models_boxplot(results_dict: dict,
                           title: str = 'Model Accuracy Comparison (5 Runs)',
                           save_dir: str = 'outputs',
                           filename: str = 'model_comparison_boxplot.png'):
    """
    Boxplot comparing multiple models across N evaluation runs.
    Shows median, IQR, and outliers — more informative than a single number.

    Args:
        results_dict : {model_name: [acc_run1, acc_run2, ...], ...}
        title        : Plot title.
        save_dir     : Output directory.
        filename     : Saved filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    df      = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_dict.items()]))
    melted  = df.melt(var_name='Model', value_name='Accuracy')

    plt.figure(figsize=(max(10, len(results_dict) * 1.4), 6))
    sns.boxplot(
        x='Model', y='Accuracy', data=melted,
        palette='Set2', width=0.5, linewidth=1.5
    )
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
# 6. Summary Table  (printed at end of each script)
# ─────────────────────────────────────────────────────────────────────

def print_summary_table(results: list, title: str = 'MODEL RESULTS SUMMARY'):
    """
    Print a formatted summary table of all models evaluated in the script.
    Called once at the very end of each script.

    Args:
        results : List of dicts, each with keys:
                  'Model', 'Best_Acc', 'Mean_Acc', 'Std_Acc'
                  (Std_Acc optional — omitted for single-run models)
        title   : Table header string.
    """
    df = pd.DataFrame(results)
    df = df.sort_values('Best_Acc', ascending=False).reset_index(drop=True)

    # Format percentages
    for col in ['Best_Acc', 'Mean_Acc']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")
    if 'Std_Acc' in df.columns:
        df['Std_Acc'] = df['Std_Acc'].apply(lambda x: f"±{x*100:.2f}%")

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}\n")

    return df


def save_summary_csv(results: list, filename: str, save_dir: str = 'outputs'):
    """Save the summary results list to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    path = os.path.join(save_dir, filename)
    df.to_csv(path, index=False)
    print(f"  Saved summary CSV: {path}")


# ─────────────────────────────────────────────────────────────────────
# 7. Model Persistence
# ─────────────────────────────────────────────────────────────────────

def save_model(model, model_name: str, save_dir: str = 'outputs'):
    """Save trained Keras model in .keras format."""
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    path = os.path.join(save_dir, f"{safe_name}.keras")
    model.save(path)
    print(f"  Model saved: {path}")
    return path
