"""
experiment_utils.py
Shared utilities for experiment validation framework.
"""

import os
import random
import time

import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from Eigenface_f import Eigenface_f

# ── iShip-1 class names ────────────────────────────────────────────────
_ISHIP1_CLASS_NAMES = [
    'Bulk Carrier',
    'Cargo Ship',
    'Other Ship',
    'Passenger Ship',
    'Fishing Vessel',
    'Pleasure Craft',
]


# ══════════════════════════════════════════════════════════════════════
# Dataset Loaders
# ══════════════════════════════════════════════════════════════════════

def load_seaships(
    dataset_path: str,
    image_size: tuple[int, int],
    samples_per_class: int,
) -> np.ndarray:
    """
    Load the seaships_gray dataset.

    Parameters
    ----------
    dataset_path      : path to the seaships_gray folder
    image_size        : (width, height) for cv2.resize
    samples_per_class : number of samples per class (1-indexed filenames)

    Returns
    -------
    All_DAT_raw : np.ndarray, shape (DIM, Class_NUM * samples_per_class)
                  float64, values in [0, 1]
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f'seaships dataset path not found: {dataset_path}')

    width, height = image_size
    DIM = width * height
    CLASS_NUM = 6
    All_DAT_raw = np.zeros((DIM, CLASS_NUM * samples_per_class), dtype=np.float64)

    col = 0
    for r in range(1, CLASS_NUM + 1):
        prefix = f'ship_0{r}' if r < 10 else f'ship_{r}'
        for t in range(1, samples_per_class + 1):
            filepath = os.path.join(dataset_path, f'{prefix}_{t}.jpg')
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Warning: cannot read {filepath}')
                col += 1
                continue
            img = cv2.resize(img, (width, height))
            All_DAT_raw[:, col] = img.astype(np.float64).ravel() / 255.0
            col += 1

    return All_DAT_raw


def load_iship1(
    dataset_root: str,
    image_size: tuple[int, int],
    samples_per_class: int,
    random_seed: int = 1234,
) -> np.ndarray:
    """
    Load the iShip-1 dataset by cropping bounding boxes from YOLO annotations.

    Parameters
    ----------
    dataset_root      : root folder containing images/ and labels/ subfolders
    image_size        : (width, height) for cv2.resize
    samples_per_class : number of samples to collect per class
    random_seed       : seed for reproducible shuffling

    Returns
    -------
    All_DAT_raw : np.ndarray, shape (DIM, Class_NUM * samples_per_class)
                  float64, values in [0, 1]
    """
    labels_dir = os.path.join(dataset_root, 'labels')
    images_dir = os.path.join(dataset_root, 'images')

    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f'Invalid iShip-1 dataset path: {dataset_root}\n'
            f'Expected subfolders: labels/ and images/'
        )

    CLASS_NUM = len(_ISHIP1_CLASS_NAMES)
    class_samples: list[list[np.ndarray]] = [[] for _ in range(CLASS_NUM)]

    random.seed(random_seed)
    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith('.txt'))

    for label_filename in label_files:
        if all(len(s) >= samples_per_class for s in class_samples):
            break

        label_path = os.path.join(labels_dir, label_filename)
        image_path = os.path.join(images_dir, os.path.splitext(label_filename)[0] + '.jpg')

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Warning: cannot read {image_path}')
            continue

        h, w = image.shape
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id < 0 or class_id >= CLASS_NUM:
                continue
            if len(class_samples[class_id]) >= samples_per_class:
                continue

            xc, yc, wr, hr = map(float, parts[1:])
            x1 = max(0, int(round((xc - wr / 2) * w)))
            y1 = max(0, int(round((yc - hr / 2) * h)))
            x2 = min(w, int(round((xc + wr / 2) * w)))
            y2 = min(h, int(round((yc + hr / 2) * h)))
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)

            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, image_size, interpolation=cv2.INTER_LINEAR)
            class_samples[class_id].append(crop.astype(np.float64).ravel() / 255.0)

    for idx, samples in enumerate(class_samples):
        if len(samples) < samples_per_class:
            raise ValueError(
                f'Class {idx} ({_ISHIP1_CLASS_NAMES[idx]}) has only {len(samples)} samples; '
                f'{samples_per_class} required.'
            )
        random.shuffle(samples)
        class_samples[idx] = samples[:samples_per_class]

    width, height = image_size
    DIM = width * height
    All_DAT_raw = np.zeros((DIM, CLASS_NUM * samples_per_class), dtype=np.float64)
    for class_id, samples in enumerate(class_samples):
        for sample_idx, vec in enumerate(samples):
            All_DAT_raw[:, class_id * samples_per_class + sample_idx] = vec

    return All_DAT_raw


# ══════════════════════════════════════════════════════════════════════
# Data Preparation (split + PCA + 3D reshape)
# ══════════════════════════════════════════════════════════════════════

def prepare_data(
    All_DAT_raw: np.ndarray,
    Class_NUM: int,
    Class_Sample_NUM: int,
    train_ratio: float,
    max_disc: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Split into train/test, run PCA (Eigenface), reshape to 3D.

    Parameters
    ----------
    All_DAT_raw      : shape (DIM, Class_NUM * Class_Sample_NUM)
    Class_NUM        : number of classes
    Class_Sample_NUM : total samples per class
    train_ratio      : fraction in (0, 1) or percentage in (1, 100)
    max_disc         : maximum PCA dimensions (capped internally)

    Returns
    -------
    Train_SET_3D     : shape (Disc_NUM, Class_Train_NUM, Class_NUM)
    Test_SET_3D      : shape (Disc_NUM, Class_Test_NUM,  Class_NUM)
    disc_set         : shape (DIM, Disc_NUM)  — PCA basis (for noise experiments)
    Class_Train_NUM  : int
    Class_Test_NUM   : int
    """
    # Normalise ratio to fraction
    ratio = train_ratio / 100.0 if train_ratio > 1.0 else train_ratio

    Class_Train_NUM = max(1, round(Class_Sample_NUM * ratio))
    Class_Test_NUM = Class_Sample_NUM - Class_Train_NUM

    DIM = All_DAT_raw.shape[0]
    Train_DAT = np.zeros((DIM, Class_Train_NUM * Class_NUM), dtype=np.float64)
    Test_DAT  = np.zeros((DIM, Class_Test_NUM  * Class_NUM), dtype=np.float64)

    s_tr = s_te = 0
    for r in range(Class_NUM):
        block = All_DAT_raw[:, r * Class_Sample_NUM: (r + 1) * Class_Sample_NUM]
        Train_DAT[:, s_tr: s_tr + Class_Train_NUM] = block[:, :Class_Train_NUM]
        Test_DAT [:, s_te: s_te + Class_Test_NUM ] = block[:, Class_Train_NUM:]
        s_tr += Class_Train_NUM
        s_te += Class_Test_NUM

    Disc_NUM = min(max_disc, Class_NUM * Class_Train_NUM - 1)
    disc_set, _ = Eigenface_f(Train_DAT, Disc_NUM)

    Train_SET_PCA = disc_set.T @ Train_DAT   # (Disc_NUM, Train_NUM)
    Test_SET_PCA  = disc_set.T @ Test_DAT    # (Disc_NUM, Test_NUM)

    Train_SET_3D = Train_SET_PCA.reshape(Disc_NUM, Class_Train_NUM, Class_NUM, order='F')
    Test_SET_3D  = Test_SET_PCA .reshape(Disc_NUM, Class_Test_NUM,  Class_NUM, order='F')

    return Train_SET_3D, Test_SET_3D, disc_set, Class_Train_NUM, Class_Test_NUM


# ══════════════════════════════════════════════════════════════════════
# Boundary Sample Identification
# ══════════════════════════════════════════════════════════════════════

def identify_boundary_samples(
    Train_SET_3D: np.ndarray,
    Test_SET_3D: np.ndarray,
    ratio_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Identify boundary test samples in PCA feature space.

    A test sample is considered a boundary sample when it lies near the
    decision boundary between classes, defined as:

        d_same / (d_diff + eps) ∈ [ratio_range[0], ratio_range[1]]

    where:
        d_same = distance to the nearest same-class training sample
        d_diff = distance to the nearest different-class training sample

    Parameters
    ----------
    Train_SET_3D : shape (DIM, Class_Train_NUM, Class_NUM)
    Test_SET_3D  : shape (DIM, Class_Test_NUM,  Class_NUM)
    ratio_range  : (low, high) inclusive bounds for the boundary criterion

    Returns
    -------
    is_boundary : bool ndarray, shape (Class_NUM * Class_Test_NUM,)
        True for each test sample that satisfies the boundary criterion.
        Ordering matches np.repeat(np.arange(Class_NUM), Class_Test_NUM)
        (i.e. column-major / Fortran order over the 3D array).
    """
    DIM, Class_Train_NUM, Class_NUM = Train_SET_3D.shape
    _,   Class_Test_NUM,  _         = Test_SET_3D.shape
    eps = 1e-10
    low, high = ratio_range

    # Flatten to 2D column arrays, column-major so class grouping is preserved
    # Train: (DIM, Class_NUM * Class_Train_NUM)
    Train_2D = Train_SET_3D.reshape(DIM, Class_NUM * Class_Train_NUM, order='F')
    # Test:  (DIM, Class_NUM * Class_Test_NUM)
    Test_2D  = Test_SET_3D .reshape(DIM, Class_NUM * Class_Test_NUM,  order='F')

    # True class label for each training column (0-based)
    train_labels = np.repeat(np.arange(Class_NUM), Class_Train_NUM)
    # True class label for each test column (0-based)
    test_labels  = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

    n_test  = Class_NUM * Class_Test_NUM
    is_boundary = np.zeros(n_test, dtype=bool)

    for i in range(n_test):
        test_vec = Test_2D[:, i]          # (DIM,)
        true_cls = test_labels[i]

        # Squared Euclidean distances to all training samples
        diff = Train_2D - test_vec[:, np.newaxis]   # (DIM, N_train)
        sq_dists = np.sum(diff ** 2, axis=0)         # (N_train,)

        same_mask = (train_labels == true_cls)
        diff_mask = ~same_mask

        if not np.any(same_mask) or not np.any(diff_mask):
            # Cannot determine boundary without both classes present
            continue

        d_same = np.sqrt(np.min(sq_dists[same_mask]))
        d_diff = np.sqrt(np.min(sq_dists[diff_mask]))

        ratio = d_same / (d_diff + eps)
        is_boundary[i] = (low <= ratio <= high)

    return is_boundary


# ══════════════════════════════════════════════════════════════════════
# Noise Injection Utilities
# ══════════════════════════════════════════════════════════════════════

def add_gaussian_noise(
    image_vec: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to a normalised image vector.

    Parameters
    ----------
    image_vec : 1D float64 array, values in [0, 1]
    sigma     : standard deviation of the noise (Noise_Level)
    rng       : numpy random Generator for reproducibility

    Returns
    -------
    Noisy copy of image_vec, clipped to [0, 1], same shape and dtype.
    """
    noise = rng.normal(loc=0.0, scale=sigma, size=image_vec.shape)
    return np.clip(image_vec + noise, 0.0, 1.0).astype(image_vec.dtype)


def add_salt_pepper_noise(
    image_vec: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add salt-and-pepper noise to a normalised image vector.

    Total corrupted pixels = round(2 * ratio * N).
    Half are set to 0.0 (pepper), half to 1.0 (salt).

    Parameters
    ----------
    image_vec : 1D float64 array, values in [0, 1]
    ratio     : corruption fraction per polarity
    rng       : numpy random Generator for reproducibility

    Returns
    -------
    Noisy copy of image_vec, same shape and dtype.
    """
    out = image_vec.copy()
    N = len(out)
    total_corrupted = round(2 * ratio * N)
    n_each = total_corrupted // 2  # pepper count == salt count

    indices = rng.choice(N, size=total_corrupted, replace=False)
    out[indices[:n_each]] = 0.0   # pepper
    out[indices[n_each:]] = 1.0   # salt
    return out


def add_occlusion(
    image_vec: np.ndarray,
    area_ratio: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Zero out a square region centred in the image.

    Parameters
    ----------
    image_vec    : 1D float64 array, values in [0, 1]
    area_ratio   : fraction of total pixels to occlude
    image_shape  : (height, width) of the original image

    Returns
    -------
    Copy of image_vec with the centre square zeroed, same shape and dtype.
    """
    out = image_vec.copy()
    h, w = image_shape
    side = int(round((area_ratio * h * w) ** 0.5))
    side = min(side, h, w)

    r0 = (h - side) // 2
    c0 = (w - side) // 2
    img_2d = out.reshape(h, w)
    img_2d[r0: r0 + side, c0: c0 + side] = 0.0
    return img_2d.ravel()


def apply_noise_to_test_dat(
    Test_DAT_raw_2d: np.ndarray,
    noise_fn,
    noise_level: float,
    image_shape: tuple[int, int],
    seed: int = 42,
) -> np.ndarray:
    """
    Apply a noise function to every test sample in pixel space.

    Parameters
    ----------
    Test_DAT_raw_2d : shape (DIM, N_test_total), float64, values in [0, 1]
    noise_fn        : one of add_gaussian_noise, add_salt_pepper_noise,
                      add_occlusion (called as noise_fn(vec, noise_level, ...))
    noise_level     : Noise_Level parameter forwarded to noise_fn
    image_shape     : (height, width) — required by add_occlusion; ignored
                      by the other two functions but always passed for a
                      uniform call signature
    seed            : random seed for reproducibility

    Returns
    -------
    Noisy copy of Test_DAT_raw_2d, same shape and dtype.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(Test_DAT_raw_2d)
    for i in range(Test_DAT_raw_2d.shape[1]):
        vec = Test_DAT_raw_2d[:, i]
        if noise_fn is add_occlusion:
            out[:, i] = noise_fn(vec, noise_level, image_shape)
        else:
            out[:, i] = noise_fn(vec, noise_level, rng)
    return out


# ══════════════════════════════════════════════════════════════════════
# Classifier Runner
# ══════════════════════════════════════════════════════════════════════

def run_all_classifiers(
    Train_SET_3D: np.ndarray,
    Test_SET_3D: np.ndarray,
    true_labels: np.ndarray,
    enabled_methods: dict,
    lam: float = 0.05,
    K_knn: int = 5,
    sci_fsnc_kwargs: dict | None = None,
) -> dict:
    """
    Run all enabled classifiers and return per-method metrics.

    Parameters
    ----------
    Train_SET_3D    : shape (DIM, Class_Train_NUM, Class_NUM)
    Test_SET_3D     : shape (DIM, Class_Test_NUM,  Class_NUM)
    true_labels     : 1D int array, shape (Class_NUM * Class_Test_NUM,), 0-based
    enabled_methods : dict mapping method name → bool, e.g.
                      {'MDC': False, 'KNN': False, 'FKNN': True,
                       'SRC': True, 'FSNC': True, 'SCI_FSNC': True}
                      SCI_FSNC, FSNC, SRC are always run regardless of flag.
    lam             : L1 regularisation parameter (default 0.05)
    K_knn           : K for KNN / FKNN (default 5)
    sci_fsnc_kwargs : optional dict of extra keyword arguments forwarded to
                      Classifier_SCI_FSNC only (e.g. K_MIN, sci_threshold).
                      When None, defaults inside Classifier_SCI_FSNC are used.

    Returns
    -------
    results : dict[str, dict]
        {
          'SCI_FSNC': {'acc': float, 'prec': float, 'rec': float,
                       'f1': float, 'time': float, 'cm': np.ndarray},
          ...
        }
    """
    from Classifier_SCI_FSNC import Classifier_SCI_FSNC
    from Classifier_FSNC import Classifier_FSNC
    from Classifier_SRC import Classifier_SRC
    from Classifier_fuzzy_KNN import Classifier_fuzzy_KNN
    from Classifier_kNNC import Classifier_kNNC
    from Classifier_MDC import Classifier_MDC

    _, Class_Test_NUM, Class_NUM = Test_SET_3D.shape
    n_test = Class_NUM * Class_Test_NUM

    results = {}

    def _metrics(pred_labels_2d: np.ndarray, cm: np.ndarray, elapsed: float) -> dict:
        """Compute accuracy + macro precision/recall/F1 from flat predictions."""
        pred_flat = pred_labels_2d.ravel()  # row-major (C order) matches true_labels layout
        acc = float(np.sum(pred_flat == true_labels) / n_test)
        prec = float(precision_score(true_labels, pred_flat, average='macro', zero_division=0))
        rec  = float(recall_score(true_labels, pred_flat, average='macro', zero_division=0))
        f1   = float(f1_score(true_labels, pred_flat, average='macro', zero_division=0))
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'time': elapsed, 'cm': cm, 'pred_labels': pred_labels_2d}

    # ── SCI_FSNC (always enabled) ──────────────────────────────────────
    try:
        kwargs = dict(sci_fsnc_kwargs) if sci_fsnc_kwargs else {}
        # Allow SCI-FSNC to use its own lam if specified in sci_fsnc_kwargs,
        # otherwise fall back to the shared lam parameter.
        sci_lam = kwargs.pop('lam', lam)
        Miss, Pred, CM, tc = Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, sci_lam, **kwargs)
        results['SCI_FSNC'] = _metrics(Pred, CM, tc)
    except Exception as e:
        print(f'Warning: SCI_FSNC failed — {e}')

    # ── FSNC (always enabled) ──────────────────────────────────────────
    try:
        Miss, Pred, CM, tc = Classifier_FSNC(Train_SET_3D, Test_SET_3D, lam)
        results['FSNC'] = _metrics(Pred, CM, tc)
    except Exception as e:
        print(f'Warning: FSNC failed — {e}')

    # ── SRC (optional) ────────────────────────────────────────────────
    if enabled_methods.get('SRC', True):
        try:
            t0 = time.perf_counter()
            Miss, Pred, CM = Classifier_SRC(Train_SET_3D, Test_SET_3D, lam)
            tc = time.perf_counter() - t0
            results['SRC'] = _metrics(Pred, CM, tc)
        except Exception as e:
            print(f'Warning: SRC failed — {e}')

    # ── FKNN (optional) ────────────────────────────────────────────────
    if enabled_methods.get('FKNN', False):
        try:
            t0 = time.perf_counter()
            Miss, Pred = Classifier_fuzzy_KNN(Train_SET_3D, Test_SET_3D, K_knn)
            tc = time.perf_counter() - t0
            # fuzzy_KNN does not return CM — build it from predictions
            pred_flat = Pred.ravel()  # C order, matches true_labels layout
            cm = np.zeros((Class_NUM, Class_NUM), dtype=int)
            for true_c in range(Class_NUM):
                for pred_c in range(Class_NUM):
                    cm[true_c, pred_c] = int(np.sum(
                        (true_labels == true_c) & (pred_flat == pred_c)
                    ))
            results['FKNN'] = _metrics(Pred, cm, tc)
        except Exception as e:
            print(f'Warning: FKNN failed — {e}')

    # ── KNN (optional; time measured externally) ───────────────────────
    if enabled_methods.get('KNN', False):
        try:
            t0 = time.perf_counter()
            Miss, Pred, CM = Classifier_kNNC(Train_SET_3D, Test_SET_3D, K_knn)
            tc = time.perf_counter() - t0
            results['KNN'] = _metrics(Pred, CM, tc)
        except Exception as e:
            print(f'Warning: KNN failed — {e}')

    # ── MDC (optional; time measured externally) ───────────────────────
    if enabled_methods.get('MDC', False):
        try:
            t0 = time.perf_counter()
            Miss, Pred, CM = Classifier_MDC(Train_SET_3D, Test_SET_3D)
            tc = time.perf_counter() - t0
            results['MDC'] = _metrics(Pred, CM, tc)
        except Exception as e:
            print(f'Warning: MDC failed — {e}')

    return results


# ══════════════════════════════════════════════════════════════════════
# Fixed-K SCI-FSNC Runner
# ══════════════════════════════════════════════════════════════════════

class FixedK_SCI_FSNC_Runner:
    """
    Run SCI-FSNC with a fixed K value by temporarily monkey-patching
    _compute_adaptive_K in the Classifier_SCI_FSNC module.

    The original function is always restored in a finally block so that
    subsequent calls to the standard SCI-FSNC are unaffected.

    Parameters
    ----------
    fixed_k : int
        The constant K value to use instead of the adaptive selection.
    """

    def __init__(self, fixed_k: int) -> None:
        self.fixed_k = fixed_k

    def run(
        self,
        Train_SET_3D: np.ndarray,
        Test_SET_3D: np.ndarray,
        lam: float,
    ) -> tuple:
        """
        Run SCI-FSNC with fixed_k as the neighbourhood size.

        Parameters
        ----------
        Train_SET_3D : shape (DIM, Class_Train_NUM, Class_NUM)
        Test_SET_3D  : shape (DIM, Class_Test_NUM,  Class_NUM)
        lam          : L1 regularisation parameter

        Returns
        -------
        Same 4-tuple as Classifier_SCI_FSNC:
            (Miss_NUM, Predict_Labels, CM, Time_Cost)
        """
        import Classifier_SCI_FSNC as mod

        fixed_k = self.fixed_k
        original_fn = mod._compute_adaptive_K

        # Replace with a lambda that always returns fixed_k
        mod._compute_adaptive_K = lambda w, sci, K_MIN=3, threshold=1e-5: fixed_k

        try:
            result = mod.Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, lam)
        finally:
            # Always restore the original function
            mod._compute_adaptive_K = original_fn

        return result


# ══════════════════════════════════════════════════════════════════════
# Plotting Utilities
# ══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Consistent colour and display-name mapping across all figures
METHOD_COLORS: dict[str, str] = {
    'SCI_FSNC': 'green',
    'FSNC':     'red',
    'SRC':      'orange',
    'FKNN':     'blue',
    'KNN':      'purple',
    'MDC':      'brown',
}

METHOD_LABELS: dict[str, str] = {
    'SCI_FSNC': 'SCI-FSNC (Proposed)',
    'FSNC':     'FSNC',
    'SRC':      'SRC',
    'FKNN':     'FKNN',
    'KNN':      'KNN',
    'MDC':      'MDC',
}

# Display order (most important first)
_METHOD_ORDER = ['SCI_FSNC', 'FSNC', 'SRC', 'FKNN', 'KNN', 'MDC']


def _ordered_methods(available: list[str]) -> list[str]:
    """Return methods sorted by canonical display order."""
    ordered = [m for m in _METHOD_ORDER if m in available]
    extras  = [m for m in available if m not in ordered]
    return ordered + extras


def plot_accuracy_vs_ratio(
    results_by_ratio: dict,
    train_ratios: list[int],
    dataset_name: str,
    save_path: str,
) -> None:
    """
    Line plot: accuracy vs training ratio for each method.

    Parameters
    ----------
    results_by_ratio : dict[int, dict[str, dict]]
        Outer key = train ratio (%), inner key = method name,
        value = ClassifierResult dict with at least 'acc'.
    train_ratios     : list of training ratios (%) in ascending order
    dataset_name     : used in the figure title
    save_path        : full path where the PNG is saved
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # Collect methods present in any ratio
    all_methods: set[str] = set()
    for r in train_ratios:
        all_methods.update(results_by_ratio.get(r, {}).keys())
    methods = _ordered_methods(list(all_methods))

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in methods:
        accs = [results_by_ratio.get(r, {}).get(method, {}).get('acc', None)
                for r in train_ratios]
        valid_x = [train_ratios[i] for i, a in enumerate(accs) if a is not None]
        valid_y = [a for a in accs if a is not None]
        if not valid_y:
            continue
        ax.plot(
            valid_x, valid_y,
            marker='o',
            color=METHOD_COLORS.get(method, None),
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel('Training Ratio (%)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy vs Training Ratio — {dataset_name}', fontsize=13)
    ax.set_xticks(train_ratios)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(
    results_at_ratio: dict,
    dataset_name: str,
    train_ratio: int,
    save_path: str,
) -> None:
    """
    Grouped bar chart: Accuracy / Precision / Recall / F1 per method.

    Parameters
    ----------
    results_at_ratio : dict[str, dict]
        Key = method name, value = ClassifierResult dict.
    dataset_name     : used in the figure title
    train_ratio      : training ratio (%) used in the figure title
    save_path        : full path where the PNG is saved
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    methods = _ordered_methods(list(results_at_ratio.keys()))
    metrics = ['acc', 'prec', 'rec', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(methods):
        values = [results_at_ratio[method].get(m, 0.0) for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, values,
            width=bar_width,
            color=METHOD_COLORS.get(method, None),
            label=METHOD_LABELS.get(method, method),
            alpha=0.85,
        )

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        f'Classification Metrics at {train_ratio}% Training — {dataset_name}',
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_robustness_curve(
    noise_levels: list,
    acc_by_method: dict,
    noise_type: str,
    dataset_name: str,
    save_path: str,
) -> None:
    """
    Line plot: accuracy vs noise level for each method.

    Parameters
    ----------
    noise_levels  : list of noise level values (x-axis)
    acc_by_method : dict[str, list[float]]
        Key = method name, value = accuracy at each noise level.
    noise_type    : 'gaussian', 'salt_pepper', or 'occlusion'
    dataset_name  : used in the figure title
    save_path     : full path where the PNG is saved
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    noise_display = {
        'gaussian':    'Gaussian Noise (σ)',
        'salt_pepper': 'Salt & Pepper Noise (ratio)',
        'occlusion':   'Occlusion (area ratio)',
    }
    x_label = noise_display.get(noise_type, f'{noise_type} level')

    methods = _ordered_methods(list(acc_by_method.keys()))

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in methods:
        accs = acc_by_method.get(method, [])
        if not accs:
            continue
        ax.plot(
            noise_levels, accs,
            marker='o',
            color=METHOD_COLORS.get(method, None),
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(
        f'Robustness — {noise_type.replace("_", " ").title()} — {dataset_name}',
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_adaptive_k_bar(
    k_values: list,
    overall_acc: dict,
    boundary_acc: dict,
    dataset_name: str,
    save_path: str,
) -> None:
    """
    Grouped bar chart: overall accuracy and boundary-sample accuracy
    for adaptive K vs each fixed K value.

    Parameters
    ----------
    k_values     : list of fixed K values, e.g. [3, 5, 7, 10, 15]
    overall_acc  : dict mapping strategy label → overall accuracy float.
                   Strategy labels: 'Adaptive' and str(k) for each fixed k.
    boundary_acc : dict mapping strategy label → boundary accuracy float.
    dataset_name : used in the figure title
    save_path    : full path where the PNG is saved
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # Build ordered x-axis labels: Adaptive first, then fixed K values
    labels = ['Adaptive'] + [str(k) for k in k_values]
    x = np.arange(len(labels))
    bar_width = 0.35

    overall_vals  = [overall_acc.get(lbl, 0.0)  for lbl in labels]
    boundary_vals = [boundary_acc.get(lbl, 0.0) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - bar_width / 2, overall_vals,  bar_width,
           label='Overall Accuracy',          color='steelblue',  alpha=0.85)
    ax.bar(x + bar_width / 2, boundary_vals, bar_width,
           label='Boundary-Sample Accuracy',  color='darkorange', alpha=0.85)

    ax.set_xlabel('K Strategy', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Adaptive K vs Fixed K — {dataset_name}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# ships_dataset Loader & Prepare
# ══════════════════════════════════════════════════════════════════════

_SHIPS_CLASS_NAMES = [
    'Aircraft Carrier', 'Bulkers', 'Car Carrier', 'Container Ship',
    'Cruise', 'DDG', 'Recreational', 'Sailboat', 'Submarine', 'Tug',
]


def load_ships_split(
    split_dir: str,
    image_size: tuple[int, int],
    samples_per_class: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one split (train or test) of ships_dataset.

    Parameters
    ----------
    split_dir         : path to the split folder (e.g. datasets/ships_dataset/train)
    image_size        : (width, height) for cv2.resize
    samples_per_class : how many samples to use per class (capped at available)
    random_seed       : for reproducible shuffling

    Returns
    -------
    X : np.ndarray, shape (DIM, Class_NUM * samples_per_class), float64 in [0,1]
    y : np.ndarray, shape (Class_NUM * samples_per_class,), int  0-based labels
    """
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'ships_dataset split not found: {split_dir}')

    rng = np.random.default_rng(random_seed)
    width, height = image_size
    DIM = width * height
    CLASS_NUM = len(_SHIPS_CLASS_NAMES)

    all_vecs: list[np.ndarray] = []
    all_labels: list[int] = []

    for cls_idx, cls_name in enumerate(_SHIPS_CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        files = sorted(
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpeg', '.jpg', '.png'))
        )
        rng.shuffle(files)
        files = files[:samples_per_class]

        for fname in files:
            img = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            all_vecs.append(img.astype(np.float64).ravel() / 255.0)
            all_labels.append(cls_idx)

    X = np.array(all_vecs, dtype=np.float64).T   # (DIM, N)
    y = np.array(all_labels, dtype=int)
    return X, y


def prepare_ships_data(
    dataset_root: str,
    image_size: tuple[int, int] = (64, 64),
    samples_per_class_train: int = 300,
    samples_per_class_test: int = 100,
    max_disc: int = 80,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, np.ndarray]:
    """
    Load ships_dataset (pre-split train/test), run PCA, reshape to 3D.

    Parameters
    ----------
    dataset_root            : path to ships_dataset folder
    image_size              : (width, height)
    samples_per_class_train : samples per class from train split
    samples_per_class_test  : samples per class from test split
    max_disc                : max PCA dimensions
    random_seed             : reproducibility seed

    Returns
    -------
    Train_SET_3D     : (Disc_NUM, samples_per_class_train, CLASS_NUM)
    Test_SET_3D      : (Disc_NUM, samples_per_class_test,  CLASS_NUM)
    disc_set         : (DIM, Disc_NUM)
    Class_Train_NUM  : int
    Class_Test_NUM   : int
    Class_NUM        : int
    Test_DAT_raw_2d  : (DIM, Class_Test_NUM * CLASS_NUM), float64 raw pixel data
    """
    CLASS_NUM = len(_SHIPS_CLASS_NAMES)

    X_train, _ = load_ships_split(
        os.path.join(dataset_root, 'train'),
        image_size, samples_per_class_train, random_seed,
    )
    X_test, _ = load_ships_split(
        os.path.join(dataset_root, 'test'),
        image_size, samples_per_class_test, random_seed,
    )

    # X_train shape: (DIM, CLASS_NUM * samples_per_class_train)
    # Columns are ordered: all class-0 samples, then class-1, ...
    # load_ships_split iterates classes in order so this holds.

    Disc_NUM = min(max_disc, CLASS_NUM * samples_per_class_train - 1)
    disc_set, _ = Eigenface_f(X_train, Disc_NUM)

    Train_PCA = disc_set.T @ X_train   # (Disc_NUM, CLASS_NUM * n_train)
    Test_PCA  = disc_set.T @ X_test    # (Disc_NUM, CLASS_NUM * n_test)

    Train_SET_3D = Train_PCA.reshape(Disc_NUM, samples_per_class_train, CLASS_NUM, order='F')
    Test_SET_3D  = Test_PCA .reshape(Disc_NUM, samples_per_class_test,  CLASS_NUM, order='F')

    return Train_SET_3D, Test_SET_3D, disc_set, samples_per_class_train, samples_per_class_test, CLASS_NUM, X_test


# ══════════════════════════════════════════════════════════════════════
# Annotation Visualisation
# ══════════════════════════════════════════════════════════════════════

def plot_annotation_viz(
    test_images_raw: np.ndarray,
    true_labels: np.ndarray,
    predict_labels_2d: np.ndarray,
    class_names: list,
    image_shape: tuple,
    dataset_name: str,
    method_name: str,
    save_path: str,
    max_per_class: int = 20,
) -> None:
    """
    Generate a grid PNG of annotated test images (YOLO-style bounding boxes).

    Parameters
    ----------
    test_images_raw   : (DIM, N_test_total), float64 [0, 1], column-major class order
    true_labels       : (N_test_total,), int 0-based, matches column order of test_images_raw
    predict_labels_2d : (Class_NUM, Class_Test_NUM), int 0-based (classifier output)
    class_names       : list of str, length = Class_NUM
    image_shape       : (height, width) of each image
    dataset_name      : used in figure title and filename
    method_name       : used in figure title and filename
    save_path         : full path where the PNG is saved
    max_per_class     : max samples to show per class (default 20)
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    Class_NUM, Class_Test_NUM = predict_labels_2d.shape
    h, w = image_shape

    # Flatten predict_labels_2d to 1D in column-major order to match true_labels
    pred_flat = predict_labels_2d.flatten(order='F')  # (Class_NUM * Class_Test_NUM,)

    # Collect up to max_per_class annotated images per class
    annotated: list[np.ndarray] = []
    counts = np.zeros(Class_NUM, dtype=int)

    for i in range(len(true_labels)):
        true_c = true_labels[i]
        if counts[true_c] >= max_per_class:
            continue

        # Reconstruct uint8 BGR image
        img_gray = (test_images_raw[:, i].reshape(h, w) * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        pred_c = pred_flat[i]
        correct = (pred_c == true_c)
        color = (0, 255, 0) if correct else (0, 0, 255)  # BGR: green / red

        # Full-image bounding box (2px border)
        cv2.rectangle(img_bgr, (1, 1), (w - 2, h - 2), color, thickness=2)

        # Predicted class label at top-left
        label_text = class_names[pred_c] if pred_c < len(class_names) else str(pred_c)
        cv2.putText(
            img_bgr, label_text,
            (3, max(10, h // 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, color, 1, cv2.LINE_AA,
        )

        annotated.append(img_bgr)
        counts[true_c] += 1

    if not annotated:
        print(f'Warning: plot_annotation_viz — no images to display for {dataset_name}/{method_name}')
        return

    n_imgs = len(annotated)
    cols = min(10, max_per_class)
    rows = (n_imgs + cols - 1) // cols

    # Build grid canvas
    grid_h = rows * h
    grid_w = cols * w
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 200  # light-grey background

    for idx, img in enumerate(annotated):
        r, c = divmod(idx, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    # Convert BGR → RGB for matplotlib
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(max(8, cols * w / 40), max(4, rows * h / 40)))
    ax.imshow(canvas_rgb)
    ax.axis('off')
    ax.set_title(
        f'Annotation Viz — {dataset_name} — {METHOD_LABELS.get(method_name, method_name)}',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
