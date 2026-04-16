"""
exp3_adaptive_k.py
Experiment 3: Adaptive K effectiveness — compare SCI-FSNC (adaptive K)
against fixed-K variants, with boundary-sample accuracy breakdown.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import os
import numpy as np

import Classifier_SCI_FSNC as sci_fsnc_mod
from experiment_utils import (
    load_seaships,
    load_iship1,
    prepare_data,
    identify_boundary_samples,
    FixedK_SCI_FSNC_Runner,
    plot_adaptive_k_bar,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATASET      = 'seaships'   # 'seaships' or 'iship1'
TRAIN_RATIO  = 70           # fixed training ratio (%)
FIXED_K_VALUES = [3, 5, 7, 10, 15]
LAM          = 0.05

DATASET_CONFIGS = {
    'seaships': {
        'loader':            'seaships',
        'dataset_path':      os.path.join('datasets', 'seaships_gray'),
        'image_size':        (64, 64),
        'samples_per_class': 300,
        'class_num':         6,
    },
    'iship1': {
        'loader':            'iship1',
        'dataset_root':      os.path.join('datasets', 'iShip-1'),
        'image_size':        (64, 64),
        'samples_per_class': 300,
        'class_num':         6,
    },
}

RESULT_DIR = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def _run_adaptive_with_k_tracking(
    Train_SET_3D: np.ndarray,
    Test_SET_3D: np.ndarray,
    lam: float,
) -> tuple:
    """
    Run standard SCI-FSNC while capturing every K_dynamic value chosen
    by _compute_adaptive_K across all test samples.

    Strategy: temporarily wrap _compute_adaptive_K to log each return value,
    then restore the original function in a finally block.

    Parameters
    ----------
    Train_SET_3D : shape (DIM, Class_Train_NUM, Class_NUM)
    Test_SET_3D  : shape (DIM, Class_Test_NUM,  Class_NUM)
    lam          : L1 regularisation parameter

    Returns
    -------
    result       : 4-tuple (Miss_NUM, Predict_Labels, CM, Time_Cost)
    k_log        : list[int] — K_dynamic value recorded for each test sample
    """
    k_log: list[int] = []
    original_fn = sci_fsnc_mod._compute_adaptive_K

    def _tracking_wrapper(w_y, sci_val, K_MIN=3, threshold=1e-5):
        k = original_fn(w_y, sci_val, K_MIN=K_MIN, threshold=threshold)
        k_log.append(k)
        return k

    sci_fsnc_mod._compute_adaptive_K = _tracking_wrapper
    try:
        result = sci_fsnc_mod.Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, lam)
    finally:
        sci_fsnc_mod._compute_adaptive_K = original_fn

    return result, k_log


def _accuracy_on_mask(
    Predict_Labels: np.ndarray,
    true_labels: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute accuracy restricted to the samples indicated by mask.

    Parameters
    ----------
    Predict_Labels : shape (Class_NUM, Class_Test_NUM), 0-based int
    true_labels    : shape (Class_NUM * Class_Test_NUM,), 0-based int
    mask           : bool array, shape (Class_NUM * Class_Test_NUM,)

    Returns
    -------
    Accuracy as a float in [0, 1], or 0.0 if no samples match the mask.
    """
    pred_flat = Predict_Labels.ravel()  # row-major (C order) matches true_labels layout
    if not np.any(mask):
        return 0.0
    return float(np.sum(pred_flat[mask] == true_labels[mask]) / np.sum(mask))


def run_adaptive_k_experiment(dataset_name: str, cfg: dict) -> None:
    """Run the full adaptive-K experiment for one dataset."""
    print(f'\n{"#" * 60}')
    print(f'# Adaptive K Experiment — Dataset: {dataset_name}')
    print(f'{"#" * 60}')

    # ── Load data ────────────────────────────────────────────────────────
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

    # ── Prepare data (split + PCA) ───────────────────────────────────────
    Train_SET_3D, Test_SET_3D, _disc_set, Class_Train_NUM, Class_Test_NUM = prepare_data(
        All_DAT_raw=All_DAT_raw,
        Class_NUM=Class_NUM,
        Class_Sample_NUM=Class_Sample_NUM,
        train_ratio=TRAIN_RATIO,
    )

    true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)
    n_test = Class_NUM * Class_Test_NUM

    print(f'Train samples/class: {Class_Train_NUM}, Test samples/class: {Class_Test_NUM}')
    print(f'Total test samples: {n_test}')

    # ── Task 4.1: Identify boundary samples ─────────────────────────────
    print('\nIdentifying boundary samples...')
    is_boundary = identify_boundary_samples(Train_SET_3D, Test_SET_3D)
    n_boundary = int(np.sum(is_boundary))
    print(f'Boundary samples: {n_boundary} / {n_test} ({n_boundary/n_test*100:.1f}%)')

    if n_boundary == 0:
        print('Warning: no boundary samples found — boundary accuracy will be skipped.')

    # ── Task 4.3: Run adaptive SCI-FSNC with K_dynamic tracking ─────────
    print('\nRunning adaptive SCI-FSNC (with K tracking)...')
    (Miss_adapt, Pred_adapt, CM_adapt, tc_adapt), k_log = _run_adaptive_with_k_tracking(
        Train_SET_3D, Test_SET_3D, LAM
    )

    # Req 3.5: print mean and std of K_dynamic across all test samples
    k_arr = np.array(k_log, dtype=float)
    k_mean = float(np.mean(k_arr))
    k_std  = float(np.std(k_arr))
    print(f'K_dynamic stats — mean: {k_mean:.3f}, std: {k_std:.3f}  '
          f'(min={int(k_arr.min())}, max={int(k_arr.max())})')

    overall_acc_adaptive  = float((n_test - Miss_adapt) / n_test)
    boundary_acc_adaptive = _accuracy_on_mask(Pred_adapt, true_labels, is_boundary)

    print(f'Adaptive K — Overall Acc: {overall_acc_adaptive*100:.2f}%  '
          f'Boundary Acc: {boundary_acc_adaptive*100:.2f}%  '
          f'Time: {tc_adapt:.2f}s')

    # ── Task 4.2: Run fixed-K variants ───────────────────────────────────
    overall_acc:  dict[str, float] = {'Adaptive': overall_acc_adaptive}
    boundary_acc: dict[str, float] = {'Adaptive': boundary_acc_adaptive}

    for k in FIXED_K_VALUES:
        print(f'\nRunning fixed K={k}...')
        runner = FixedK_SCI_FSNC_Runner(fixed_k=k)
        Miss_k, Pred_k, CM_k, tc_k = runner.run(Train_SET_3D, Test_SET_3D, LAM)

        ov_acc = float((n_test - Miss_k) / n_test)
        bd_acc = _accuracy_on_mask(Pred_k, true_labels, is_boundary)

        overall_acc[str(k)]  = ov_acc
        boundary_acc[str(k)] = bd_acc

        print(f'  Fixed K={k} — Overall Acc: {ov_acc*100:.2f}%  '
              f'Boundary Acc: {bd_acc*100:.2f}%  Time: {tc_k:.2f}s')

    # ── Summary table ────────────────────────────────────────────────────
    print(f'\n{"─" * 55}')
    print(f'{"Strategy":<12}  {"Overall Acc":>12}  {"Boundary Acc":>13}')
    print(f'{"─" * 55}')
    for label in ['Adaptive'] + [str(k) for k in FIXED_K_VALUES]:
        ov = overall_acc.get(label, 0.0)
        bd = boundary_acc.get(label, 0.0)
        print(f'{label:<12}  {ov*100:>11.2f}%  {bd*100:>12.2f}%')
    print(f'{"─" * 55}')

    # ── Task 4.4: Save plot ──────────────────────────────────────────────
    save_path = os.path.join(RESULT_DIR, f'adaptive_k_comparison_{dataset_name}.png')
    plot_adaptive_k_bar(
        k_values=FIXED_K_VALUES,
        overall_acc=overall_acc,
        boundary_acc=boundary_acc,
        dataset_name=dataset_name,
        save_path=save_path,
    )
    print(f'\nSaved: {save_path}')


def main() -> None:
    cfg = DATASET_CONFIGS[DATASET]
    run_adaptive_k_experiment(DATASET, cfg)
    print('\nExperiment 3 complete.')


if __name__ == '__main__':
    main()
