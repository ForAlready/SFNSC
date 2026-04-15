"""
main_experiment.py
主程序：融合稀疏表示的自适应模糊分类器验证评估
包含三种方法对比：
1. 模糊 KNN (FKNN) - Baseline
2. 传统 FSNC - Baseline
3. 融合 SCI 动态优化的 SCI-FSNC - Proposed

数据集: seaships — 水面舰船目标图像
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Eigenface_f import Eigenface_f
from Classifier_fuzzy_KNN import Classifier_fuzzy_KNN
from Classifier_FSNC import Classifier_FSNC
from Classifier_SCI_FSNC import Classifier_SCI_FSNC  # <-- 引入新的优化算法

# ─── 中文字体配置（可选，如系统有中文字体则生效）────────────────────────
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════
# 1. 数据集加载
# ═══════════════════════════════════════════════════════════════════════
dataset_type = 'seaships'
print(f'Loading dataset: {dataset_type} ...')

Image_row_NUM    = 64  # 图像高度 
Image_column_NUM = 64  # 图像宽度
NN               = Image_row_NUM * Image_column_NUM
Class_NUM        = 6
Class_Sample_NUM = 300   # 每个类的总样本数

All_DAT_raw = np.zeros((NN, Class_NUM * Class_Sample_NUM))
s = 0   # 0-based 列索引

for r in range(1, Class_NUM + 1):
    for t in range(1, Class_Sample_NUM + 1):
        if r < 10:
            filepath = os.path.join('.', 'datasets/seaships_gray', f'ship_0{r}_{t}.jpg')
        else:
            filepath = os.path.join('.', 'datasets/seaships_gray', f'ship_{r}_{t}.jpg')

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: File not found or cannot read {filepath}")
            s += 1
            continue

        img = cv2.resize(img, (Image_column_NUM, Image_row_NUM))
        B = img.astype(np.float64) / 255.0
        All_DAT_raw[:, s] = B.ravel()
        s += 1

# ═══════════════════════════════════════════════════════════════════════
# 2. 超参数设定
# ═══════════════════════════════════════════════════════════════════════
Train_Ratio_List = [70, 80]
K_FKNN  = 5
lam     = 0.05
# SCI-FSNC 的动态邻域边界
K_MIN = 3


# 结果记录数组
Results_FKNN_Acc = np.zeros(len(Train_Ratio_List))
Results_FKNN_Prec = np.zeros(len(Train_Ratio_List))
Results_FKNN_Rec  = np.zeros(len(Train_Ratio_List))
Results_FKNN_F1   = np.zeros(len(Train_Ratio_List))
Results_FSNC_Acc = np.zeros(len(Train_Ratio_List))
Results_FSNC_Prec = np.zeros(len(Train_Ratio_List))
Results_FSNC_Rec  = np.zeros(len(Train_Ratio_List))
Results_FSNC_F1   = np.zeros(len(Train_Ratio_List))
Results_SCI_Acc  = np.zeros(len(Train_Ratio_List))
Results_SCI_Prec = np.zeros(len(Train_Ratio_List))
Results_SCI_Rec  = np.zeros(len(Train_Ratio_List))
Results_SCI_F1   = np.zeros(len(Train_Ratio_List))

Time_FKNN        = np.zeros(len(Train_Ratio_List))  
Time_FSNC        = np.zeros(len(Train_Ratio_List))
Time_SCI         = np.zeros(len(Train_Ratio_List))

CM_SCI_last = None

# ═══════════════════════════════════════════════════════════════════════
# 3. 对比实验主体循环
# ═══════════════════════════════════════════════════════════════════════
for idx, ratio in enumerate(Train_Ratio_List):
    Class_Train_NUM = max(1, round(Class_Sample_NUM * ratio / 100))
    Class_Test_NUM  = Class_Sample_NUM - Class_Train_NUM

    print(f'\n--- Processing Train Ratio: {ratio}% '
          f'({Class_Train_NUM} Train / {Class_Test_NUM} Test per class) ---')

    # ── 分割训练集与测试集 ──────────────────────────────────────────────
    Train_DAT = np.zeros((NN, Class_Train_NUM * Class_NUM))
    Test_DAT  = np.zeros((NN, Class_Test_NUM  * Class_NUM))
    s_tr = 0
    s_te = 0

    for r in range(Class_NUM):
        class_samples = All_DAT_raw[:, r * Class_Sample_NUM : (r + 1) * Class_Sample_NUM]
        Train_DAT[:, s_tr : s_tr + Class_Train_NUM] = class_samples[:, :Class_Train_NUM]
        Test_DAT [:, s_te : s_te + Class_Test_NUM ] = class_samples[:, Class_Train_NUM:]
        s_tr += Class_Train_NUM
        s_te += Class_Test_NUM

    # ── PCA 特征提取 (Eigenface) ────────────────────────────────────────
    Disc_NUM = min(80, Class_NUM * Class_Train_NUM - 1)
    disc_set, _ = Eigenface_f(Train_DAT, Disc_NUM)

    Train_SET_PCA = disc_set.T @ Train_DAT   # (Disc_NUM, Train_NUM)
    Test_SET_PCA  = disc_set.T @ Test_DAT    # (Disc_NUM, Test_NUM)

    # 重塑为 3D: (Disc_NUM, Samples_Per_Class, Class_NUM)
    Train_SET_3D = Train_SET_PCA.reshape(Disc_NUM, Class_Train_NUM, Class_NUM, order='F')
    Test_SET_3D  = Test_SET_PCA .reshape(Disc_NUM, Class_Test_NUM,  Class_NUM, order='F')

    Total_Test_NUM = Class_Test_NUM * Class_NUM
    true_labels = np.repeat(np.arange(Class_NUM), Class_Test_NUM)

    # ── 1. FKNN (Baseline) ──────────────────────────────────────────────
    t0 = time.perf_counter()
    Miss_NUM_FKNN, pred_fk = Classifier_fuzzy_KNN(Train_SET_3D, Test_SET_3D, K_FKNN)
    Time_FKNN[idx] = time.perf_counter() - t0
    pred_fk_flat = pred_fk.ravel()
    Results_FKNN_Acc[idx] = accuracy_score(true_labels, pred_fk_flat)
    Results_FKNN_Prec[idx] = precision_score(true_labels, pred_fk_flat, average='macro', zero_division=0)
    Results_FKNN_Rec[idx] = recall_score(true_labels, pred_fk_flat, average='macro', zero_division=0)
    Results_FKNN_F1[idx]  = f1_score(true_labels, pred_fk_flat, average='macro', zero_division=0)

    # ── 2. 原版 FSNC (Baseline) ─────────────────────────────────────────
    Miss_NUM_FSNC, pred_fs, _, Time_FSNC[idx] = \
        Classifier_FSNC(Train_SET_3D, Test_SET_3D, lam)
    pred_fs_flat = pred_fs.ravel()
    Results_FSNC_Acc[idx] = accuracy_score(true_labels, pred_fs_flat)
    Results_FSNC_Prec[idx] = precision_score(true_labels, pred_fs_flat, average='macro', zero_division=0)
    Results_FSNC_Rec[idx] = recall_score(true_labels, pred_fs_flat, average='macro', zero_division=0)
    Results_FSNC_F1[idx]  = f1_score(true_labels, pred_fs_flat, average='macro', zero_division=0)

    # ── 3. SCI-FSNC (Proposed) ──────────────────────────────────────────
    Miss_NUM_SCI, pred_sci, CM_SCI, Time_SCI[idx] = \
        Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, lam=lam)
    pred_sci_flat = pred_sci.ravel()
    Results_SCI_Acc[idx] = accuracy_score(true_labels, pred_sci_flat)
    Results_SCI_Prec[idx] = precision_score(true_labels, pred_sci_flat, average='macro', zero_division=0)
    Results_SCI_Rec[idx] = recall_score(true_labels, pred_sci_flat, average='macro', zero_division=0)
    Results_SCI_F1[idx]  = f1_score(true_labels, pred_sci_flat, average='macro', zero_division=0)

    print(f'  FKNN      Accuracy: {Results_FKNN_Acc[idx]*100:.2f}% | Prec: {Results_FKNN_Prec[idx]*100:.2f}% | Rec: {Results_FKNN_Rec[idx]*100:.2f}% | F1: {Results_FKNN_F1[idx]*100:.2f}% | Time: {Time_FKNN[idx]:.2f} s')
    print(f'  FSNC      Accuracy: {Results_FSNC_Acc[idx]*100:.2f}% | Prec: {Results_FSNC_Prec[idx]*100:.2f}% | Rec: {Results_FSNC_Rec[idx]*100:.2f}% | F1: {Results_FSNC_F1[idx]*100:.2f}% | Time: {Time_FSNC[idx]:.2f} s')
    print(f'  SCI-FSNC  Accuracy: {Results_SCI_Acc[idx]*100:.2f}% | Prec: {Results_SCI_Prec[idx]*100:.2f}% | Rec: {Results_SCI_Rec[idx]*100:.2f}% | F1: {Results_SCI_F1[idx]*100:.2f}% | Time: {Time_SCI[idx]:.2f} s')

    # 最后一个比例保留混淆矩阵(提取最优算法的)
    if idx == len(Train_Ratio_List) - 1:
        CM_SCI_last = CM_SCI
        Class_Train_NUM_last = Class_Train_NUM
        Class_Test_NUM_last  = Class_Test_NUM

# ─── 绘制混淆矩阵 ────────────────────────────────────────────────────
if CM_SCI_last is not None:
    print('\n=== Plotting SCI-FSNC Confusion Matrix ===')
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(CM_SCI_last, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=True)

    ax.set_title(
        f'SCI-FSNC Confusion Matrix\n'
        f'(Total: {Class_Sample_NUM} | Train: {Class_Train_NUM_last} | Test: {Class_Test_NUM_last})',
        fontsize=13
    )
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)

    tick_marks = np.arange(Class_NUM)
    ax.set_xticks(tick_marks + 0.5)
    ax.set_yticks(tick_marks + 0.5)
    ax.set_xticklabels([str(c + 1) for c in range(Class_NUM)])
    ax.set_yticklabels([str(c + 1) for c in range(Class_NUM)])

    plt.tight_layout()
    plt.savefig('SCI_FSNC_confusion_matrix.png', dpi=150)

# ═══════════════════════════════════════════════════════════════════════
# 4. 可视化指标对比图
# ═══════════════════════════════════════════════════════════════════════
# 4.1 准确率对比
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(Train_Ratio_List, Results_FKNN_Acc * 100, '-bs',
        linewidth=2, markersize=8, label='Fuzzy KNN')
ax.plot(Train_Ratio_List, Results_FSNC_Acc * 100, '-ro',
        linewidth=2, markersize=8, label='FSNC')
ax.plot(Train_Ratio_List, Results_SCI_Acc * 100, '-g^',  # 添加绿色三角表示我们的新算法
        linewidth=2, markersize=8, label='SCI-FSNC (Proposed)')

ax.set_xlabel('Percentage of Training Samples per Class (%)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Comparison: FKNN vs FSNC vs SCI-FSNC')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('accuracy_comparison_3methods.png', dpi=150)

# 4.2 时间开销对比
x = np.arange(len(Train_Ratio_List))
width = 0.25  # 缩小柱子宽度以容纳3个柱状图

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, Time_FKNN, width, label='Fuzzy KNN', color='blue', alpha=0.7)
ax.bar(x,         Time_FSNC, width, label='FSNC', color='red', alpha=0.7)
ax.bar(x + width, Time_SCI,  width, label='SCI-FSNC (Proposed)', color='green', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in Train_Ratio_List])
ax.set_xlabel('Percentage of Training Samples per Class (%)')
ax.set_ylabel('Running Time (Seconds)')
ax.set_title('Computational Time Comparison (3 Methods)')
ax.legend(loc='upper left')
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('time_comparison_3methods.png', dpi=150)

print("\n实验完成！所有对比图表均已保存为图片。")
