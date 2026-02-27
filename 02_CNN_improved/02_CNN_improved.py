"""
02_CNN_improved.py
==================
IMPROVED CNN: Normalization, BatchNorm, Augmentation, 3-way split, 5-run protocol.
Compatible with: TF 2.19.1 | Keras 3.13.2 | NumPy 2.0.2
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gc
import keras
from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          Dense, Flatten, Concatenate, BatchNormalization)
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from utils.data_utils import (load_data, split_data, augment_training_data,
                               to_channel_inputs, make_dataset)
from utils.eval_utils import (evaluate_model, plot_training_history,
                               compare_models_boxplot, print_summary_table,
                               save_summary_csv, save_model)

# ── Configuration ────────────────────────────────────────────────────
MAT_PATH     = os.path.join(os.path.dirname(__file__), '..', 'Turbine.mat')
SAVE_DIR     = os.path.join(os.path.dirname(__file__), 'outputs')
EPOCHS       = 100
BATCH_SIZE   = 64
N_RUNS       = 5
RANDOM_STATE = 42

os.makedirs(SAVE_DIR, exist_ok=True)
summary_results = []
run_accuracies  = []

print("=" * 60)
print("  02 - CNN IMPROVED")
print("=" * 60)

# ── Step 1: Load & Normalize ──────────────────────────────────────────
print("Step 1: Loading and normalizing data...")
X, Y_onehot, y_int = load_data(MAT_PATH, normalize=True)
print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")

(X_train_full, X_val_full, X_test,
 y_train_full, y_val_full, y_test,
 yi_train_full, yi_val_full, yi_test) = split_data(
    X, Y_onehot, y_int, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE
)

X_test_ch = to_channel_inputs(X_test)
test_ds   = make_dataset(X_test_ch, y_test, batch_size=BATCH_SIZE, shuffle_data=False)
print(f"\n  Test set fixed: {len(X_test)} samples.")

# ── Step 2: Model Builder ─────────────────────────────────────────────
def build_improved_model(n_classes: int = 7):
    def build_branch(input_shape=(1000, 1)):
        inp = Input(shape=input_shape)
        x   = Conv1D(64, 14, activation='relu')(inp)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Conv1D(64, 14, activation='relu')(x)
        x   = BatchNormalization()(x)
        x   = MaxPooling1D(2)(x)
        x   = Dropout(0.5)(x)
        x   = Flatten()(x)
        return inp, x

    branches, model_inputs = [], []
    for _ in range(3):
        inp, out = build_branch()
        model_inputs.append(inp)
        branches.append(out)

    merged = Concatenate()(branches)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(n_classes, activation='softmax')(merged)
    model  = Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ── Step 3: 5-Run Training Loop ───────────────────────────────────────
print(f"\nStep 2: {N_RUNS}-run training protocol...")

best_acc, best_model, best_hist = 0, None, None

for run in range(N_RUNS):
    print(f"\n{'─'*55}\n  Run {run+1}/{N_RUNS}\n{'─'*55}")

    keras.backend.clear_session()
    gc.collect()

    X_tr, X_v, yi_tr, yi_v = train_test_split(
        np.vstack([X_train_full, X_val_full]),
        np.concatenate([yi_train_full, yi_val_full]),
        test_size=0.15/0.85,
        stratify=np.concatenate([yi_train_full, yi_val_full]),
        random_state=run
    )
    y_tr = to_categorical(yi_tr, num_classes=7)
    y_v  = to_categorical(yi_v,  num_classes=7)

    X_tr_aug, y_tr_aug = augment_training_data(X_tr, y_tr, noise=True, shift=True, gain=True)

    train_ds = make_dataset(to_channel_inputs(X_tr_aug), y_tr_aug,
                            batch_size=BATCH_SIZE, shuffle_data=True)
    val_ds   = make_dataset(to_channel_inputs(X_v), y_v,
                            batch_size=BATCH_SIZE, shuffle_data=False)

    model = build_improved_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks, verbose=1)

    plot_training_history(history, model_name=f'CNN Improved - Run {run+1}', save_dir=SAVE_DIR)

    run_acc, _ = evaluate_model(
        model=model, X_test=test_ds,
        y_test_onehot=y_test, y_test_int=yi_test,
        model_name=f'CNN Improved - Run {run+1}',
        save_dir=SAVE_DIR, is_branch_input=True
    )
    run_accuracies.append(run_acc)
    print(f"  Run {run+1} accuracy: {run_acc*100:.2f}%")

    if run_acc > best_acc:
        best_acc, best_model, best_hist = run_acc, model, history
        print(f"  New best: {best_acc*100:.2f}%")

# ── Step 4: Summary ───────────────────────────────────────────────────
print(f"\n  Best : {best_acc*100:.2f}%")
print(f"  Mean : {np.mean(run_accuracies)*100:.2f}%")
print(f"  Std  : {np.std(run_accuracies)*100:.2f}%")

compare_models_boxplot({'CNN Improved': run_accuracies},
                       title='CNN Improved - 5 Runs',
                       save_dir=SAVE_DIR,
                       filename='CNN_Improved_variance_boxplot.png')

summary_results.append({
    'Model': 'CNN Improved', 'Best_Acc': best_acc,
    'Mean_Acc': np.mean(run_accuracies), 'Std_Acc': np.std(run_accuracies),
    'Runs': N_RUNS, 'Notes': '3-way split, norm, BatchNorm, aug, EarlyStopping, 5-run'
})

save_model(best_model, 'CNN_Improved_best', save_dir=SAVE_DIR)
print_summary_table(summary_results, title='CNN IMPROVED - RESULTS SUMMARY')
save_summary_csv(summary_results, 'CNN_Improved_results.csv', save_dir=SAVE_DIR)

print(f"\n  -> Next: 03_CNN_multiscale_final.py")
