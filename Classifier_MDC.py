"""
Classifier_MDC.py
最小距离分类器 (Minimum Distance Classifier, MDC)

决策规则: 将测试样本分配到与其类均值距离最小的类别。
"""

import numpy as np


def Classifier_MDC(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
):
    """
    最小距离分类器

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (DIM, Class_Train_NUM, Class_NUM)
    Test_DAT  : np.ndarray, shape (DIM, Class_Test_NUM,  Class_NUM)

    Returns
    -------
    Miss_NUM       : int
    Predict_Labels : np.ndarray, shape (Class_NUM, Class_Test_NUM) — 0-based
    CM             : np.ndarray, shape (Class_NUM, Class_NUM)
    """
    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    # 计算每个类别的均值向量，shape (DIM, Class_NUM)
    Class_Means = Train_DAT.mean(axis=1)   # mean over axis=1 (samples per class)

    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)
    Miss_NUM = 0
    CM = np.zeros((Class_NUM, Class_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            test_sample = Test_DAT[:, j, i]   # shape (DIM,)

            # 计算与每个类均值的欧氏距离
            diff = Class_Means - test_sample[:, np.newaxis]   # (DIM, Class_NUM)
            dists = np.sum(diff ** 2, axis=0)                 # (Class_NUM,)

            predict_idx = int(np.argmin(dists))
            Predict_Labels[i, j] = predict_idx
            CM[i, predict_idx] += 1
            if predict_idx != i:
                Miss_NUM += 1

    return Miss_NUM, Predict_Labels, CM
