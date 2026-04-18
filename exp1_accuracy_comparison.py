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
    prepare_ships_data,
    run_all_classifiers,
    plot_accuracy_vs_ratio,
    plot_metrics_bar,
    plot_annotation_viz,
    _SHIPS_CLASS_NAMES,
)

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_RATIOS = [50, 60, 70, 80]   # training ratios (%)
K_KNN  = 5                         # K for KNN / FKNN

ENABLED_METHODS = {
    'MDC':      False,   # optional: Minimum Distance Classifier
    'KNN':      True,    # optional: standard KNN
    'FKNN':     False,   # optional: Fuzzy KNN
    'SRC':      False,   # required: Sparse Representation Classifier
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
        # Classifier hyperparameters (tuned for seaships)
        'lam':             0.05,
        'max_disc':        80,
        'sci_fsnc_kwargs': {'K_MIN': 3, 'sci_threshold': 1e-4},
    },
    'iship1': {
        'loader':          'iship1',
        'dataset_root':    os.path.join('datasets', 'iShip-1'),
        'image_size':      (64, 64),
        'samples_per_class': 300,
        'class_num':       6,
        # Shared hyperparameters (apply to all classifiers)
        # lam=0.2 is required for iShip-1: smaller values cause SRC/FSNC to
        # produce near-zero sparse coefficients on this dataset's PCA features.
        'lam':             0.2,
        'max_disc':        80,
        # SCI-FSNC-only hyperparameters (tuned via hyperparameter_tuning.py)
        'sci_fsnc_kwargs': {'K_MIN': 1, 'sci_threshold': 0.01},
    },
    'ships_dataset': {
        'loader':                    'ships',
        'dataset_root':              os.path.join('datasets', 'ships_dataset'),
        'image_size':                (64, 64),
        'samples_per_class_train':   300,
        'samples_per_class_test':    100,
        'class_num':                 10,
        # Classifier hyperparameters
        'lam':             0.05,
        'max_disc':        80,
        'sci_fsnc_kwargs': {'K_MIN': 3, 'sci_threshold': 1e-4},
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

    # ── ships_dataset: pre-split, no ratio loop ───────────────────────
    if cfg['loader'] == 'ships':
        (Train_SET_3D, Test_SET_3D, _disc_set,
         Class_Train_NUM, Class_Test_NUM, Class_NUM,
         Test_DAT_raw_2d) = prepare_ships_data(
            dataset_root=cfg['dataset_root'],
            image_size=cfg['image_size'],
            samples_per_class_train=cfg['samples_per_class_train'],
            samples_per_class_test=cfg['samples_per_class_test'],
            max_disc=cfg.get('max_disc', 80),
        )

        true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

        print(f'\n--- ships_dataset (fixed split: {Class_Train_NUM} train / {Class_Test_NUM} test per class) ---')
        results = run_all_classifiers(
            Train_SET_3D=Train_SET_3D,
            Test_SET_3D=Test_SET_3D,
            true_labels=true_labels,
            enabled_methods=ENABLED_METHODS,
            lam=cfg.get('lam', 0.05),
            K_knn=K_KNN,
            sci_fsnc_kwargs=cfg.get('sci_fsnc_kwargs'),
        )

        for method, metrics in results.items():
            print(
                f'  {method:<12} Acc: {metrics["acc"]*100:.2f}%  '
                f'Prec: {metrics["prec"]*100:.2f}%  '
                f'Rec: {metrics["rec"]*100:.2f}%  '
                f'F1: {metrics["f1"]*100:.2f}%  '
                f'Time: {metrics["time"]:.2f}s'
            )

        # Bar chart only (no accuracy_vs_ratio for pre-split datasets)
        plot_metrics_bar(
            results_at_ratio=results,
            dataset_name=dataset_name,
            train_ratio=Class_Train_NUM,
            save_path=os.path.join(RESULT_DIR, f'accuracy_all_methods_bar_{dataset_name}.png'),
        )
        print(f'Saved: result/accuracy_all_methods_bar_{dataset_name}.png')

        # Annotation visualisation for each enabled method
        for method, metrics in results.items():
            viz_path = os.path.join(RESULT_DIR, f'viz_{dataset_name}_{method}.png')
            w, h = cfg['image_size']
            plot_annotation_viz(
                test_images_raw=Test_DAT_raw_2d,
                true_labels=true_labels,
                predict_labels_2d=metrics['pred_labels'],
                class_names=_SHIPS_CLASS_NAMES,
                image_shape=(h, w),
                dataset_name=dataset_name,
                method_name=method,
                save_path=viz_path,
            )
            print(f'Saved: result/viz_{dataset_name}_{method}.png')

        return

    # ── seaships / iship1: variable ratio loop ────────────────────────
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
            max_disc=cfg.get('max_disc', 80),
        )

        true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

        results = run_all_classifiers(
            Train_SET_3D=Train_SET_3D,
            Test_SET_3D=Test_SET_3D,
            true_labels=true_labels,
            enabled_methods=ENABLED_METHODS,
            lam=cfg.get('lam', 0.05),
            K_knn=K_KNN,
            sci_fsnc_kwargs=cfg.get('sci_fsnc_kwargs'),
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
