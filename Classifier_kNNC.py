"""
Classifier_kNNC.py
K 近邻分类器 (K-Nearest Neighbor Classifier, kNNC)

标准 KNN 算法（硬决策版本，区别于模糊 KNN）。
决策规则: 找到 K 个最近邻，以多数投票决定类别。
"""

import numpy as np


def Classifier_kNNC(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    K: int = 1,
):
    """
    标准 K 近邻分类器

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (DIM, Class_Train_NUM, Class_NUM)
    Test_DAT  : np.ndarray, shape (DIM, Class_Test_NUM,  Class_NUM)
    K         : int — 近邻数量（默认 1，即最近邻分类器 NNC）

    Returns
    -------
    Miss_NUM       : int
    Predict_Labels : np.ndarray, shape (Class_NUM, Class_Test_NUM) — 0-based
    CM             : np.ndarray, shape (Class_NUM, Class_NUM)
    """
    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    Train_NUM = Class_Train_NUM * Class_NUM
    # MATLAB column-major reshape
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')

    # 每个训练样本对应的类别标签 (0-based)
    Train_Labels = np.repeat(np.arange(Class_NUM), Class_Train_NUM)  # shape (Train_NUM,)

    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)
    Miss_NUM = 0
    CM = np.zeros((Class_NUM, Class_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            test_sample = Test_DAT[:, j, i]   # shape (DIM,)

            # 计算平方欧氏距离
            diff = Train_SET - test_sample[:, np.newaxis]
            sq_dist = np.einsum('ij,ij->j', diff, diff)   # (Train_NUM,)

            # 找 K 个最近邻
            knn_idx = np.argsort(sq_dist)[:K]
            knn_labels = Train_Labels[knn_idx]

            # 多数投票
            votes = np.bincount(knn_labels, minlength=Class_NUM)
            predict_idx = int(np.argmax(votes))

            Predict_Labels[i, j] = predict_idx
            CM[i, predict_idx] += 1
            if predict_idx != i:
                Miss_NUM += 1

    return Miss_NUM, Predict_Labels, CM
