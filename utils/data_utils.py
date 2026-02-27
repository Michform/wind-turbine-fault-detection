"""
utils/data_utils.py
-------------------
Shared data loading, preprocessing, splitting, and augmentation utilities
used across ALL scripts in this project.

Dataset: Turbine.mat
    X : (14000, 3000) → reshaped to (14000, 3, 1000)
        3 channels = X, Y, Z accelerometer axes
        1000 time points per sample @ 100kHz sampling rate (10ms window)
    Y : (14000, 7) — one-hot encoded fault class labels

Fault Classes (0–6):
    0 : Normal operation
    1 : Fault type 1
    2 : Fault type 2
    3 : Fault type 3
    4 : Fault type 4
    5 : Fault type 5
    6 : Fault type 6
"""

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# ─────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────────────

def load_data(mat_path: str, normalize: bool = True):
    """
    Load and preprocess the Turbine.mat dataset.

    Args:
        mat_path  : Path to the .mat file.
        normalize : If True, scale X to [-1, 1] using global max absolute value.
                    This is always recommended — raw accelerometer axes can have
                    very different amplitude ranges, causing gradient instability.

    Returns:
        X        : np.ndarray shape (14000, 3, 1000) — float32
        y_onehot : np.ndarray shape (14000, 7)       — one-hot labels
        y_int    : np.ndarray shape (14000,)         — integer class labels
    """
    mat      = scipy.io.loadmat(mat_path)
    X        = mat['X'].reshape(14000, 3, 1000).astype(np.float32)
    Y        = mat['Y'].astype(np.float32)
    y_int    = np.argmax(Y, axis=1)

    if normalize:
        max_abs = np.max(np.abs(X))
        X       = X / max_abs

    return X, Y, y_int


# ─────────────────────────────────────────────────────────────────────
# 2. Splits
# ─────────────────────────────────────────────────────────────────────

def split_data(X, y_onehot, y_int,
               test_size: float = 0.15,
               val_size:  float = 0.15,
               random_state: int = 42):
    """
    3-way stratified split: train / validation / test.

    Why 3-way?
        EarlyStopping should monitor val_loss, NOT test_loss.
        If the test set drives EarlyStopping, it leaks into training.
        A separate val set keeps the test set completely unseen.

    Why stratified?
        Ensures every fault class appears in the correct proportion in
        all three splits — critical for multi-class imbalance robustness.

    Default split: 70% train | 15% val | 15% test

    Args:
        X            : Features, shape (N, 3, 1000).
        y_onehot     : One-hot labels, shape (N, 7).
        y_int        : Integer labels, shape (N,).
        test_size    : Fraction for test set.
        val_size     : Fraction for validation set.
        random_state : Reproducibility seed.

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,          (one-hot)
        yi_train, yi_val, yi_test        (integer)
    """
    # Step 1: carve out test set
    X_tv, X_test, y_tv, y_test, yi_tv, yi_test = train_test_split(
        X, y_onehot, y_int,
        test_size  = test_size,
        stratify   = y_int,
        random_state = random_state
    )

    # Step 2: split remainder into train and val
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, yi_train, yi_val = train_test_split(
        X_tv, y_tv, yi_tv,
        test_size    = relative_val,
        stratify     = yi_tv,
        random_state = random_state
    )

    print(f"  Split → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Class balance check (test):  "
          f"{np.bincount(yi_test)} (should be ~uniform)")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            yi_train, yi_val, yi_test)


def split_data_2way(X, y_onehot, y_int,
                    test_size: float = 0.2,
                    random_state: int = 42):
    """
    2-way stratified split for baseline comparison only.
    NOTE: In this setup, the test set is exposed during training via
    EarlyStopping — this is the methodological flaw we document in the
    baseline notebook. Do NOT use for final reported numbers.
    """
    X_train, X_test, y_train, y_test, yi_train, yi_test = train_test_split(
        X, y_onehot, y_int,
        test_size    = test_size,
        stratify     = y_int,
        random_state = random_state
    )
    print(f"  2-way split → Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, yi_train, yi_test


# ─────────────────────────────────────────────────────────────────────
# 3. Data Augmentation  (training set only — NEVER apply to val/test)
# ─────────────────────────────────────────────────────────────────────

def add_gaussian_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Additive Gaussian noise.

    Why: Prevents memorization of exact waveforms. Simulates real-world
    sensor noise. Makes the model robust to small measurement variations
    across different turbines or installation conditions.

    Args:
        data        : Array of any shape.
        noise_level : Standard deviation of noise (relative to [-1,1] scale).
    """
    return (data + np.random.normal(0, noise_level, data.shape)).astype(np.float32)


def time_shift(sample: np.ndarray, max_shift: int = 50) -> np.ndarray:
    """
    Circular time-shift of a single sample, shape (3, 1000).

    Why: Fault signatures don't always appear at the same position within
    a 10ms window. Shift-invariance makes the model robust to when
    within the recording window a fault transient occurs.

    Args:
        sample    : Single sample, shape (3, 1000).
        max_shift : Maximum shift in time steps (± direction).
    """
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(sample, shift, axis=1)


def random_gain(sample: np.ndarray, gain_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Random amplitude scaling of a single sample.

    Why: Sensor calibration drifts between turbines and over time.
    Small gain variation teaches the model to recognize fault patterns
    regardless of absolute signal amplitude.

    Args:
        sample     : Single sample, shape (3, 1000).
        gain_range : (min, max) multiplicative scale factor.
    """
    gain = np.random.uniform(*gain_range)
    return (sample * gain).astype(np.float32)


def augment_training_data(X_train: np.ndarray,
                          y_train: np.ndarray,
                          noise: bool = True,
                          shift: bool = True,
                          gain:  bool = True) -> tuple:
    """
    Apply augmentation strategies to training data.
    Each enabled strategy appends a full copy of the dataset → up to 4× size.

    Order of operations:
        Original data always included first.
        Each strategy applied to the ORIGINAL data (not compounded).
        Final array shuffled to prevent any ordering bias.

    Args:
        X_train : Training features, shape (N, 3, 1000).
        y_train : Training labels (one-hot or integer), shape (N, ...).
        noise   : Apply Gaussian noise augmentation.
        shift   : Apply time-shift augmentation.
        gain    : Apply random gain augmentation.

    Returns:
        X_aug, y_aug : Augmented and shuffled arrays.
    """
    X_parts = [X_train]
    y_parts = [y_train]

    if noise:
        X_parts.append(add_gaussian_noise(X_train))
        y_parts.append(y_train)

    if shift:
        shifted = np.array([time_shift(s) for s in X_train])
        X_parts.append(shifted)
        y_parts.append(y_train)

    if gain:
        gained = np.array([random_gain(s) for s in X_train])
        X_parts.append(gained)
        y_parts.append(y_train)

    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    X_out, y_out = shuffle(X_out, y_out, random_state=42)

    multiplier = len(X_parts)
    print(f"  Augmentation: {len(X_train)} → {len(X_out)} samples ({multiplier}× original)")

    return X_out, y_out


# ─────────────────────────────────────────────────────────────────────
# 4. Input Format Converters
# ─────────────────────────────────────────────────────────────────────

def to_channel_inputs(X: np.ndarray) -> list:
    """
    Convert (N, 3, 1000) → list of 3 arrays each shaped (N, 1000, 1).

    Required format for multi-branch CNN where each branch receives
    one vibration axis independently. Keras multi-input models expect
    a list of arrays, one per Input() layer.

    Args:
        X : Feature array, shape (N, 3, 1000).

    Returns:
        List of 3 arrays, each (N, 1000, 1).
    """
    return [X[:, i, :].reshape(-1, 1000, 1) for i in range(3)]


def to_rnn_input(X: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3, 1000) → (N, 1000, 3).

    RNNs expect (batch, timesteps, features).
    Here: 1000 timesteps, 3 features (X/Y/Z axes) per timestep.

    Args:
        X : Feature array, shape (N, 3, 1000).

    Returns:
        Transposed array, shape (N, 1000, 3).
    """
    return X.transpose(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────
# 5. Dataset Interrogation Utilities  (used in 00_dataset_analysis.py)
# ─────────────────────────────────────────────────────────────────────

def class_distribution(y_int: np.ndarray, fault_labels: list) -> dict:
    """
    Return count and percentage for each class.

    Args:
        y_int        : Integer class labels, shape (N,).
        fault_labels : List of string class names.

    Returns:
        Dict mapping class name → {'count': int, 'pct': float}
    """
    counts = np.bincount(y_int)
    total  = len(y_int)
    return {
        fault_labels[i]: {
            'count': int(counts[i]),
            'pct':   round(counts[i] / total * 100, 2)
        }
        for i in range(len(counts))
    }


def per_axis_stats(X: np.ndarray) -> dict:
    """
    Compute descriptive statistics per vibration axis.

    Args:
        X : Feature array, shape (N, 3, 1000).

    Returns:
        Dict with axis names as keys, stats dict as values.
    """
    axes = ['X', 'Y', 'Z']
    stats = {}
    for i, ax in enumerate(axes):
        data = X[:, i, :].flatten()
        stats[ax] = {
            'mean':    float(np.mean(data)),
            'std':     float(np.std(data)),
            'min':     float(np.min(data)),
            'max':     float(np.max(data)),
            'abs_max': float(np.max(np.abs(data))),
        }
    return stats
