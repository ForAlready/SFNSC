"""
exp1_accuracy_comparison.py
Experiment 1: Classification accuracy comparison across methods and datasets.

Loops over seaships and iShip-1 datasets at training ratios [50, 60, 70, 80]%,
runs all enabled classifiers, and saves accuracy/metrics plots to result/.
"""

import os
import numpy as np

from experiment_utils import (
    load_seaships,
    load_iship1,
    prepare_data,
    run_all_classifiers,
    plot_accuracy_vs_ratio,
    plot_metrics_bar,
)

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_RATIOS = [50, 60, 70, 80]   # training ratios (%)
LAM    = 0.05                      # L1 regularisation parameter
K_KNN  = 5                         # K for KNN / FKNN

ENABLED_METHODS = {
    'MDC':      False,   # optional: Minimum Distance Classifier
    'KNN':      False,   # optional: standard KNN
    'FKNN':     True,    # optional: Fuzzy KNN
    'SRC':      True,    # required: Sparse Representation Classifier
    'FSNC':     True,    # required: Fuzzy Sparse Neighbor Classifier
    'SCI_FSNC': True,    # required: proposed method
}

DATASET_CONFIGS = {
    'seaships': {
        'loader':          'seaships',
        'dataset_path':    os.path.join('datasets', 'seaships_gray'),
        'image_size':      (64, 64),
        'samples_per_class': 300,
        'class_num':       6,
    },
    'iship1': {
        'loader':          'iship1',
        'dataset_root':    os.path.join('datasets', 'iShip-1'),
        'image_size':      (64, 64),
        'samples_per_class': 300,
        'class_num':       6,
    },
}

RESULT_DIR = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def _print_summary(dataset_name: str, results_by_ratio: dict) -> None:
    """Print a formatted summary table: method × metric at each training ratio."""
    methods_seen: set[str] = set()
    for r in TRAIN_RATIOS:
        methods_seen.update(results_by_ratio.get(r, {}).keys())
    methods = sorted(methods_seen)

    header_cols = ['Method'] + [f'{r}% Acc' for r in TRAIN_RATIOS] + ['80% Prec', '80% Rec', '80% F1']
    col_w = 18
    print(f'\n{"=" * (col_w * len(header_cols))}')
    print(f'Dataset: {dataset_name}')
    print('  '.join(f'{h:<{col_w}}' for h in header_cols))
    print('-' * (col_w * len(header_cols)))

    for method in methods:
        row = [method]
        for r in TRAIN_RATIOS:
            acc = results_by_ratio.get(r, {}).get(method, {}).get('acc', None)
            row.append(f'{acc*100:.2f}%' if acc is not None else 'N/A')
        res80 = results_by_ratio.get(80, {}).get(method, {})
        row.append(f'{res80.get("prec", 0)*100:.2f}%')
        row.append(f'{res80.get("rec",  0)*100:.2f}%')
        row.append(f'{res80.get("f1",   0)*100:.2f}%')
        print('  '.join(f'{v:<{col_w}}' for v in row))

    print(f'{"=" * (col_w * len(header_cols))}\n')


def run_dataset(dataset_name: str, cfg: dict) -> None:
    """Load data, run classifiers at all train ratios, save plots."""
    print(f'\n{"#" * 60}')
    print(f'# Dataset: {dataset_name}')
    print(f'{"#" * 60}')

    # Load raw data once
    if cfg['loader'] == 'seaships':
        All_DAT_raw = load_seaships(
            dataset_path=cfg['dataset_path'],
            image_size=cfg['image_size'],
            samples_per_class=cfg['samples_per_class'],
        )
    else:
        All_DAT_raw = load_iship1(
            dataset_root=cfg['dataset_root'],
            image_size=cfg['image_size'],
            samples_per_class=cfg['samples_per_class'],
        )

    Class_NUM        = cfg['class_num']
    Class_Sample_NUM = cfg['samples_per_class']

    results_by_ratio: dict[int, dict] = {}

    for ratio in TRAIN_RATIOS:
        print(f'\n--- Train ratio: {ratio}% ---')

        Train_SET_3D, Test_SET_3D, _disc_set, Class_Train_NUM, Class_Test_NUM = prepare_data(
            All_DAT_raw=All_DAT_raw,
            Class_NUM=Class_NUM,
            Class_Sample_NUM=Class_Sample_NUM,
            train_ratio=ratio,
        )

        true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

        results = run_all_classifiers(
            Train_SET_3D=Train_SET_3D,
            Test_SET_3D=Test_SET_3D,
            true_labels=true_labels,
            enabled_methods=ENABLED_METHODS,
            lam=LAM,
            K_knn=K_KNN,
        )

        results_by_ratio[ratio] = results

        for method, metrics in results.items():
            print(
                f'  {method:<12} Acc: {metrics["acc"]*100:.2f}%  '
                f'Prec: {metrics["prec"]*100:.2f}%  '
                f'Rec: {metrics["rec"]*100:.2f}%  '
                f'F1: {metrics["f1"]*100:.2f}%  '
                f'Time: {metrics["time"]:.2f}s'
            )

    # ── Plots ────────────────────────────────────────────────────────────
    plot_accuracy_vs_ratio(
        results_by_ratio=results_by_ratio,
        train_ratios=TRAIN_RATIOS,
        dataset_name=dataset_name,
        save_path=os.path.join(RESULT_DIR, f'accuracy_vs_ratio_{dataset_name}.png'),
    )
    print(f'Saved: result/accuracy_vs_ratio_{dataset_name}.png')

    plot_metrics_bar(
        results_at_ratio=results_by_ratio.get(80, {}),
        dataset_name=dataset_name,
        train_ratio=80,
        save_path=os.path.join(RESULT_DIR, f'accuracy_all_methods_bar_{dataset_name}.png'),
    )
    print(f'Saved: result/accuracy_all_methods_bar_{dataset_name}.png')

    _print_summary(dataset_name, results_by_ratio)


def main() -> None:
    for dataset_name, cfg in DATASET_CONFIGS.items():
        run_dataset(dataset_name, cfg)

    print('Experiment 1 complete. All plots saved to result/.')


if __name__ == '__main__':
    main()
