"""
exp_ships_dataset.py
Experiment: Classification accuracy comparison on ships_dataset.

10-class ship image dataset (pre-split 80/20 train/test).
Runs all enabled classifiers and saves plots to result/.
"""

import os
import numpy as np

from experiment_utils import (
    prepare_ships_data,
    run_all_classifiers,
    plot_metrics_bar,
    _SHIPS_CLASS_NAMES,
)

# ── Configuration ─────────────────────────────────────────────────────
DATASET_ROOT = os.path.join('datasets', 'ships_dataset')
IMAGE_SIZE   = (64, 64)

# Cap per-class samples to keep runtime reasonable;
# train split has ~536–1636 per class, test has ~135–410.
SAMPLES_TRAIN = 300   # per class from train split
SAMPLES_TEST  = 100   # per class from test split

LAM   = 0.05
K_KNN = 5

ENABLED_METHODS = {
    'MDC':      False,
    'KNN':      False,
    'FKNN':     True,
    'SRC':      True,
    'FSNC':     True,
    'SCI_FSNC': True,
}

RESULT_DIR = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
DATASET_NAME = 'ships_dataset'
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f'\n{"#" * 60}')
    print(f'# Dataset: {DATASET_NAME}  ({len(_SHIPS_CLASS_NAMES)} classes)')
    print(f'# Train: {SAMPLES_TRAIN}/class   Test: {SAMPLES_TEST}/class')
    print(f'{"#" * 60}\n')

    print('Loading and preparing data (PCA) ...')
    Train_SET_3D, Test_SET_3D, _disc, Class_Train_NUM, Class_Test_NUM, Class_NUM, _ = \
        prepare_ships_data(
            dataset_root=DATASET_ROOT,
            image_size=IMAGE_SIZE,
            samples_per_class_train=SAMPLES_TRAIN,
            samples_per_class_test=SAMPLES_TEST,
        )

    print(f'Train_SET_3D: {Train_SET_3D.shape}')
    print(f'Test_SET_3D:  {Test_SET_3D.shape}')

    true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

    print('\nRunning classifiers ...')
    results = run_all_classifiers(
        Train_SET_3D=Train_SET_3D,
        Test_SET_3D=Test_SET_3D,
        true_labels=true_labels,
        enabled_methods=ENABLED_METHODS,
        lam=LAM,
        K_knn=K_KNN,
    )

    # ── Print results ─────────────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print(f'{"Method":<12} {"Acc":>7} {"Prec":>7} {"Rec":>7} {"F1":>7} {"Time":>8}')
    print('-' * 65)
    for method, m in results.items():
        print(
            f'{method:<12} '
            f'{m["acc"]*100:>6.2f}%  '
            f'{m["prec"]*100:>6.2f}%  '
            f'{m["rec"]*100:>6.2f}%  '
            f'{m["f1"]*100:>6.2f}%  '
            f'{m["time"]:>7.2f}s'
        )
    print('=' * 65)

    # ── Plot ──────────────────────────────────────────────────────────
    bar_path = os.path.join(RESULT_DIR, f'accuracy_all_methods_bar_{DATASET_NAME}.png')
    plot_metrics_bar(
        results_at_ratio=results,
        dataset_name=DATASET_NAME,
        train_ratio=int(SAMPLES_TRAIN / (SAMPLES_TRAIN + SAMPLES_TEST) * 100),
        save_path=bar_path,
    )
    print(f'\nSaved: {bar_path}')
    print('\nExperiment complete.')


if __name__ == '__main__':
    main()
