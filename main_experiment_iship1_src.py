"""
main_experiment_iship1_src.py

使用 iShip-1 数据集对 SRC 和 FSNC 进行分类实验。

数据集格式:
  - datasets/iShip-1/images/*.jpg
  - datasets/iShip-1/labels/*.txt  (YOLO 格式)

分类标签映射:
  0: Bulk Carrier
  1: Cargo Ship
  2: Other Ship
  3: Passenger Ship
  4: Fishing Vessel
  5: Pleasure Craft
"""

import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Eigenface_f import Eigenface_f
from Classifier_SRC import Classifier_SRC
from Classifier_FSNC import Classifier_FSNC
from Classifier_SCI_FSNC import Classifier_SCI_FSNC

CLASS_NAMES = [
    'Bulk Carrier',
    'Cargo Ship',
    'Other Ship',
    'Passenger Ship',
    'Fishing Vessel',
    'Pleasure Craft',
]


def load_iship1_dataset(
    dataset_root: str,
    image_size: tuple[int, int],
    samples_per_class: int,
    random_seed: int = 1234,
):
    """从 iShip-1 数据集中读取并裁剪船只目标样本。"""
    labels_dir = os.path.join(dataset_root, 'labels')
    images_dir = os.path.join(dataset_root, 'images')

    if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f'Invalid iShip-1 dataset path: {dataset_root}\n'
            f'Expected subfolders: labels/ and images/'
        )

    class_samples: list[list[np.ndarray]] = [[] for _ in CLASS_NAMES]
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

    random.seed(random_seed)
    processed_images = 0

    for label_filename in label_files:
        if all(len(samples) >= samples_per_class for samples in class_samples):
            break

        label_path = os.path.join(labels_dir, label_filename)
        image_name = os.path.splitext(label_filename)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Warning: 无法打开图像 {image_path}')
            continue

        height, width = image.shape
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            if len(class_samples[class_id]) >= samples_per_class:
                continue

            x_center, y_center, w_rel, h_rel = map(float, parts[1:])
            x1 = int(round((x_center - w_rel / 2) * width))
            y1 = int(round((y_center - h_rel / 2) * height))
            x2 = int(round((x_center + w_rel / 2) * width))
            y2 = int(round((y_center + h_rel / 2) * height))

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(crop, image_size, interpolation=cv2.INTER_LINEAR)
            sample = crop.astype(np.float64) / 255.0
            class_samples[class_id].append(sample.ravel())

        processed_images += 1

    for idx, samples in enumerate(class_samples):
        if len(samples) < samples_per_class:
            raise ValueError(
                f'Class {idx} ({CLASS_NAMES[idx]}) only has {len(samples)} samples, '
                f'but {samples_per_class} are required.'
            )

        random.shuffle(samples)
        class_samples[idx] = samples[:samples_per_class]

    class_count = [len(samples) for samples in class_samples]
    print('Loaded iShip-1 dataset:')
    for idx, name in enumerate(CLASS_NAMES):
        print(f'  {idx}={name}: {class_count[idx]} samples')

    NN = image_size[0] * image_size[1]
    all_data = np.zeros((NN, len(CLASS_NAMES) * samples_per_class), dtype=np.float64)
    for class_id, samples in enumerate(class_samples):
        for sample_idx, sample in enumerate(samples):
            all_data[:, class_id * samples_per_class + sample_idx] = sample

    return all_data


if __name__ == '__main__':
    dataset_root = os.path.join('.', 'datasets', 'iShip-1')
    Image_column_NUM = 64
    Image_row_NUM = 64
    NN = Image_column_NUM * Image_row_NUM

    Class_NUM = len(CLASS_NAMES)
    Class_Sample_NUM = 700 
    Train_Ratio_List = [30,50,70, 80]
    lam = 0.05

    All_DAT_raw = load_iship1_dataset(
        dataset_root,
        image_size=(Image_column_NUM, Image_row_NUM),
        samples_per_class=Class_Sample_NUM,
    )

    Results_SRC_Acc = np.zeros(len(Train_Ratio_List))
    Results_SRC_Prec = np.zeros(len(Train_Ratio_List))
    Results_SRC_Rec = np.zeros(len(Train_Ratio_List))
    Results_SRC_F1 = np.zeros(len(Train_Ratio_List))
    Results_FSNC_Acc = np.zeros(len(Train_Ratio_List))
    Results_FSNC_Prec = np.zeros(len(Train_Ratio_List))
    Results_FSNC_Rec = np.zeros(len(Train_Ratio_List))
    Results_FSNC_F1 = np.zeros(len(Train_Ratio_List))
    Results_SCI_Acc = np.zeros(len(Train_Ratio_List))
    Results_SCI_Prec = np.zeros(len(Train_Ratio_List))
    Results_SCI_Rec = np.zeros(len(Train_Ratio_List))
    Results_SCI_F1 = np.zeros(len(Train_Ratio_List))
    Time_SRC = np.zeros(len(Train_Ratio_List))
    Time_FSNC = np.zeros(len(Train_Ratio_List))
    Time_SCI = np.zeros(len(Train_Ratio_List))
    CM_SCI_last = None

    for idx, ratio in enumerate(Train_Ratio_List):
        Class_Train_NUM = max(1, round(Class_Sample_NUM * ratio / 100))
        Class_Test_NUM = Class_Sample_NUM - Class_Train_NUM

        print(f'\n--- Train Ratio: {ratio}% ({Class_Train_NUM} train / {Class_Test_NUM} test per class) ---')

        Train_DAT = np.zeros((NN, Class_Train_NUM * Class_NUM), dtype=np.float64)
        Test_DAT = np.zeros((NN, Class_Test_NUM * Class_NUM), dtype=np.float64)
        s_tr = s_te = 0

        for class_id in range(Class_NUM):
            class_block = All_DAT_raw[:, class_id * Class_Sample_NUM: (class_id + 1) * Class_Sample_NUM]
            Train_DAT[:, s_tr: s_tr + Class_Train_NUM] = class_block[:, :Class_Train_NUM]
            Test_DAT[:, s_te: s_te + Class_Test_NUM] = class_block[:, Class_Train_NUM:]
            s_tr += Class_Train_NUM
            s_te += Class_Test_NUM

        Disc_NUM = min(80, Class_NUM * Class_Train_NUM - 1)
        disc_set, _ = Eigenface_f(Train_DAT, Disc_NUM)
        Train_SET_PCA = disc_set.T @ Train_DAT
        Test_SET_PCA = disc_set.T @ Test_DAT

        Train_SET_3D = Train_SET_PCA.reshape(Disc_NUM, Class_Train_NUM, Class_NUM, order='F')
        Test_SET_3D = Test_SET_PCA.reshape(Disc_NUM, Class_Test_NUM, Class_NUM, order='F')

        Total_Test_NUM = Class_Test_NUM * Class_NUM
        true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

        t0 = time.perf_counter()
        Miss_NUM_SRC, pred_src, CM_SRC = Classifier_SRC(Train_SET_3D, Test_SET_3D, lam)
        Time_SRC[idx] = time.perf_counter() - t0
        pred_src_flat = pred_src.ravel()
        Results_SRC_Acc[idx] = accuracy_score(true_labels, pred_src_flat)
        Results_SRC_Prec[idx] = precision_score(true_labels, pred_src_flat, average='macro', zero_division=0)
        Results_SRC_Rec[idx] = recall_score(true_labels, pred_src_flat, average='macro', zero_division=0)
        Results_SRC_F1[idx] = f1_score(true_labels, pred_src_flat, average='macro', zero_division=0)

        t0 = time.perf_counter()
        Miss_NUM_FSNC, pred_fs, CM_FSNC, Time_FSNC[idx] = Classifier_FSNC(Train_SET_3D, Test_SET_3D, lam)
        pred_fs_flat = pred_fs.ravel()
        Results_FSNC_Acc[idx] = accuracy_score(true_labels, pred_fs_flat)
        Results_FSNC_Prec[idx] = precision_score(true_labels, pred_fs_flat, average='macro', zero_division=0)
        Results_FSNC_Rec[idx] = recall_score(true_labels, pred_fs_flat, average='macro', zero_division=0)
        Results_FSNC_F1[idx] = f1_score(true_labels, pred_fs_flat, average='macro', zero_division=0)

        t0 = time.perf_counter()
        Miss_NUM_SCI, pred_sci, CM_SCI, Time_SCI[idx] = Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, lam)
        pred_sci_flat = pred_sci.ravel()
        Results_SCI_Acc[idx] = accuracy_score(true_labels, pred_sci_flat)
        Results_SCI_Prec[idx] = precision_score(true_labels, pred_sci_flat, average='macro', zero_division=0)
        Results_SCI_Rec[idx] = recall_score(true_labels, pred_sci_flat, average='macro', zero_division=0)
        Results_SCI_F1[idx] = f1_score(true_labels, pred_sci_flat, average='macro', zero_division=0)

        print(f'  SRC      Accuracy: {Results_SRC_Acc[idx]*100:.2f}% | Prec: {Results_SRC_Prec[idx]*100:.2f}% | Rec: {Results_SRC_Rec[idx]*100:.2f}% | F1: {Results_SRC_F1[idx]*100:.2f}% | Time: {Time_SRC[idx]:.2f}s')
        print(f'  FSNC     Accuracy: {Results_FSNC_Acc[idx]*100:.2f}% | Prec: {Results_FSNC_Prec[idx]*100:.2f}% | Rec: {Results_FSNC_Rec[idx]*100:.2f}% | F1: {Results_FSNC_F1[idx]*100:.2f}% | Time: {Time_FSNC[idx]:.2f}s')
        print(f'  SCI-FSNC Accuracy: {Results_SCI_Acc[idx]*100:.2f}% | Prec: {Results_SCI_Prec[idx]*100:.2f}% | Rec: {Results_SCI_Rec[idx]*100:.2f}% | F1: {Results_SCI_F1[idx]*100:.2f}% | Time: {Time_SCI[idx]:.2f}s')

        if idx == len(Train_Ratio_List) - 1:
            CM_SCI_last = CM_SCI
            Class_Train_NUM_last = Class_Train_NUM
            Class_Test_NUM_last = Class_Test_NUM

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Train_Ratio_List, Results_SRC_Acc * 100, '-bs', linewidth=2, markersize=8, label='SRC')
    ax.plot(Train_Ratio_List, Results_FSNC_Acc * 100, '-ro', linewidth=2, markersize=8, label='FSNC')
    ax.plot(Train_Ratio_List, Results_SCI_Acc * 100, '-g^', linewidth=2, markersize=8, label='SCI-FSNC')
    ax.set_xlabel('Training Ratio (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('iShip-1 Classification: SRC vs FSNC vs SCI-FSNC')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('iship1_accuracy_comparison_src.png', dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(Train_Ratio_List))
    width = 0.25
    ax.bar(x - width, Time_SRC, width, label='SRC', color='blue', alpha=0.7)
    ax.bar(x, Time_FSNC, width, label='FSNC', color='red', alpha=0.7)
    ax.bar(x + width, Time_SCI, width, label='SCI-FSNC', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in Train_Ratio_List])
    ax.set_xlabel('Training Ratio (%)')
    ax.set_ylabel('Time (s)')
    ax.set_title('iShip-1 Running Time Comparison')
    ax.legend()
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig('iship1_time_comparison_src.png', dpi=150)

    if CM_SCI_last is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(CM_SCI_last, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=True)
        ax.set_title(
            f'SCI-FSNC Confusion Matrix\n'
            f'(Train: {Class_Train_NUM_last}, Test: {Class_Test_NUM_last})', fontsize=13)
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        tick_marks = np.arange(Class_NUM)
        ax.set_xticks(tick_marks + 0.5)
        ax.set_yticks(tick_marks + 0.5)
        ax.set_xticklabels([str(c + 1) for c in range(Class_NUM)])
        ax.set_yticklabels([str(c + 1) for c in range(Class_NUM)])
        plt.tight_layout()
        plt.savefig('iship1_sci_fsnc_confusion_matrix_src.png', dpi=150)

    print('\n实验完成，图像已保存为: iship1_accuracy_comparison_src.png, iship1_time_comparison_src.png, iship1_sci_fsnc_confusion_matrix_src.png')