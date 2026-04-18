"""
exp2_robustness.py
Experiment 2: Robustness test under Gaussian noise, salt-and-pepper noise,
and occlusion.

Noise is applied in pixel space BEFORE PCA projection, using the PCA basis
computed once from clean training data. Results are saved to result/.
"""

import os
import numpy as np

from experiment_utils import (
    load_seaships,
    load_iship1,
    prepare_data,
    prepare_ships_data,
    run_all_classifiers,
    apply_noise_to_test_dat,
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_occlusion,
    plot_robustness_curve,
    plot_annotation_viz,
    _SHIPS_CLASS_NAMES,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATASET     = 'seaships'   # 'seaships' or 'iship1'
TRAIN_RATIO = 70           # fixed training ratio (%)
NOISE_SEED  = 42           # random seed for reproducible noise

GAUSSIAN_LEVELS    = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
SALT_PEPPER_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
OCCLUSION_LEVELS   = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

ROBUSTNESS_METHODS = {
    'KNN':      True,    # optional
    'FKNN':     False,   # optional
    'SRC':      True,    # required
    'FSNC':     True,    # required
    'SCI_FSNC': True,    # required
}

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
    'ships_dataset': {
        'loader':                    'ships',
        'dataset_root':              os.path.join('datasets', 'ships_dataset'),
        'image_size':                (64, 64),
        'samples_per_class_train':   300,
        'samples_per_class_test':    100,
        'class_num':                 10,
    },
}

RESULT_DIR = 'result'
os.makedirs(RESULT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def _project_and_reshape(
    noisy_test_raw: np.ndarray,
    disc_set: np.ndarray,
    Disc_NUM: int,
    Class_Test_NUM: int,
    Class_NUM: int,
) -> np.ndarray:
    """
    Project noisy pixel-space test data with the pre-computed PCA basis,
    then reshape to 3D format expected by classifiers.

    Parameters
    ----------
    noisy_test_raw : shape (DIM, Class_NUM * Class_Test_NUM)
    disc_set       : shape (DIM, Disc_NUM) — PCA basis from clean training data
    Disc_NUM       : number of PCA dimensions
    Class_Test_NUM : test samples per class
    Class_NUM      : number of classes

    Returns
    -------
    noisy_Test_SET_3D : shape (Disc_NUM, Class_Test_NUM, Class_NUM)
    """
    projected = disc_set.T @ noisy_test_raw          # (Disc_NUM, N_test)
    return projected.reshape(Disc_NUM, Class_Test_NUM, Class_NUM, order='F')


def run_robustness(dataset_name: str, cfg: dict) -> None:
    """Run the full robustness experiment for one dataset."""
    print(f'\n{"#" * 60}')
    print(f'# Robustness Experiment — Dataset: {dataset_name}')
    print(f'{"#" * 60}')

    # ── Load data ────────────────────────────────────────────────────────
    Class_NUM  = cfg['class_num']
    image_size = cfg['image_size']                         # (width, height)
    image_shape = (image_size[1], image_size[0])           # (height, width) for noise fns

    if cfg['loader'] == 'ships':
        # ships_dataset: pre-split train/test; prepare_ships_data returns raw test pixels
        (Train_SET_3D, _, disc_set,
         Class_Train_NUM, Class_Test_NUM, _,
         Test_DAT_raw_2d) = prepare_ships_data(
            dataset_root=cfg['dataset_root'],
            image_size=image_size,
            samples_per_class_train=cfg['samples_per_class_train'],
            samples_per_class_test=cfg['samples_per_class_test'],
        )
        class_names = list(_SHIPS_CLASS_NAMES)
    else:
        if cfg['loader'] == 'seaships':
            All_DAT_raw = load_seaships(
                dataset_path=cfg['dataset_path'],
                image_size=image_size,
                samples_per_class=cfg['samples_per_class'],
            )
        else:
            All_DAT_raw = load_iship1(
                dataset_root=cfg['dataset_root'],
                image_size=image_size,
                samples_per_class=cfg['samples_per_class'],
            )

        Class_Sample_NUM = cfg['samples_per_class']

        # ── Split + PCA (computed once from clean training data) ─────────
        Train_SET_3D, _, disc_set, Class_Train_NUM, Class_Test_NUM = prepare_data(
            All_DAT_raw=All_DAT_raw,
            Class_NUM=Class_NUM,
            Class_Sample_NUM=Class_Sample_NUM,
            train_ratio=TRAIN_RATIO,
        )

        # ── Keep raw pixel-space test data for noise injection ───────────
        Test_DAT_raw_2d = np.zeros(
            (All_DAT_raw.shape[0], Class_Test_NUM * Class_NUM), dtype=np.float64
        )
        col = 0
        for r in range(Class_NUM):
            block = All_DAT_raw[:, r * Class_Sample_NUM: (r + 1) * Class_Sample_NUM]
            test_block = block[:, Class_Train_NUM:]
            Test_DAT_raw_2d[:, col: col + Class_Test_NUM] = test_block
            col += Class_Test_NUM

        class_names = None  # not used for non-ships datasets in viz

    Disc_NUM = disc_set.shape[1]
    true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

    print(f'Train samples/class: {Class_Train_NUM}, Test samples/class: {Class_Test_NUM}')
    print(f'PCA dimensions: {Disc_NUM}')

    # ── Noise type definitions ───────────────────────────────────────────
    noise_configs = [
        ('gaussian',    add_gaussian_noise,    GAUSSIAN_LEVELS),
        ('salt_pepper', add_salt_pepper_noise, SALT_PEPPER_LEVELS),
        ('occlusion',   add_occlusion,         OCCLUSION_LEVELS),
    ]

    for noise_type, noise_fn, noise_levels in noise_configs:
        print(f'\n--- Noise type: {noise_type} ---')
        # acc_by_method[method] = list of accuracies, one per noise level
        acc_by_method: dict[str, list[float]] = {}
        last_results: dict = {}

        for level in noise_levels:
            print(f'  Level: {level:.2f}', end='  ')

            # Req 2.6: level 0.0 → use original unmodified test images
            if level == 0.0:
                noisy_test_raw = Test_DAT_raw_2d
            else:
                noisy_test_raw = apply_noise_to_test_dat(
                    Test_DAT_raw_2d=Test_DAT_raw_2d,
                    noise_fn=noise_fn,
                    noise_level=level,
                    image_shape=image_shape,
                    seed=NOISE_SEED,
                )

            # Project with pre-computed PCA basis → 3D
            noisy_Test_SET_3D = _project_and_reshape(
                noisy_test_raw, disc_set, Disc_NUM, Class_Test_NUM, Class_NUM
            )

            results = run_all_classifiers(
                Train_SET_3D=Train_SET_3D,
                Test_SET_3D=noisy_Test_SET_3D,
                true_labels=true_labels,
                enabled_methods=ROBUSTNESS_METHODS,
                lam=0.05,
                K_knn=5,
            )
            last_results = results  # keep results from highest noise level

            accs_str = []
            for method, metrics in results.items():
                acc_by_method.setdefault(method, []).append(metrics['acc'])
                accs_str.append(f'{method}: {metrics["acc"]*100:.2f}%')
            print('  '.join(accs_str))

        # ── Save robustness curve ────────────────────────────────────────
        save_path = os.path.join(
            RESULT_DIR, f'robustness_{noise_type}_{dataset_name}.png'
        )
        plot_robustness_curve(
            noise_levels=noise_levels,
            acc_by_method=acc_by_method,
            noise_type=noise_type,
            dataset_name=dataset_name,
            save_path=save_path,
        )
        print(f'Saved: {save_path}')

        # ── Annotation visualisation (ships_dataset only, highest noise level) ──
        if cfg['loader'] == 'ships' and class_names is not None:
            for method, metrics in last_results.items():
                viz_path = os.path.join(
                    RESULT_DIR,
                    f'viz_{dataset_name}_{method}_robustness_{noise_type}.png',
                )
                plot_annotation_viz(
                    test_images_raw=Test_DAT_raw_2d,
                    true_labels=true_labels,
                    predict_labels_2d=metrics['pred_labels'],
                    class_names=class_names,
                    image_shape=image_shape,
                    dataset_name=dataset_name,
                    method_name=method,
                    save_path=viz_path,
                )
                print(f'Saved viz: {viz_path}')


def main() -> None:
    cfg = DATASET_CONFIGS[DATASET]
    run_robustness(DATASET, cfg)
    print('\nExperiment 2 complete. All robustness plots saved to result/.')


if __name__ == '__main__':
    main()
