"""
Classifier_fuzzy_KNN.py
模糊 K 近邻分类器 (Fuzzy KNN)

算法来源: James M. Keller et al., 1985
步骤:
  1. 对训练集计算模糊隶属度矩阵 U
  2. 对测试样本使用模糊 KNN 分类
"""

import numpy as np


def _k_min(distances: np.ndarray, K: int):
    """
    找到距离数组中最小的 K 个值及其原始索引。

    Parameters
    ----------
    distances : np.ndarray, shape (N,)
    K         : int

    Returns
    -------
    min_vals  : np.ndarray, shape (K,)
    min_idx   : np.ndarray, shape (K,)  — 0-based 索引
    """
    idx = np.argsort(distances)[:K]
    return distances[idx], idx


def Classifier_fuzzy_KNN(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    K: int,
) -> int:
    """
    模糊 K 近邻分类器

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (DIM, Class_Train_NUM, Class_NUM)
    Test_DAT  : np.ndarray, shape (DIM, Class_Test_NUM,  Class_NUM)
    K         : int — 近邻数量

    Returns
    -------
    Miss_NUM  : int — 分类错误的样本总数
    """
    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    # 重塑为 2D: (DIM, Train_NUM)
    Train_NUM = Class_Train_NUM * Class_NUM
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')
    # MATLAB reshape 默认按列优先 (Fortran/column-major) 展开

    # =========================================================
    # 步骤 1: 计算模糊隶属度矩阵 U — shape (Class_NUM, Train_NUM)
    # =========================================================
    U = np.zeros((Class_NUM, Train_NUM))

    for s in range(Class_NUM):
        for t in range(Class_Train_NUM):
            n = s * Class_Train_NUM + t  # 样本在 Train_SET 中的全局 0-based 索引

            sample = Train_DAT[:, t, s]  # shape (DIM,)

            # 计算与所有训练样本的平方欧氏距离
            diff = Train_SET - sample[:, np.newaxis]   # (DIM, Train_NUM)
            sq_dist = np.einsum('ij,ij->j', diff, diff)  # (Train_NUM,)

            # 找 K+1 个最近邻（含自己）
            _, knn_idx = _k_min(sq_dist, K + 1)

            # 统计每类的近邻个数（跳过第一个，即自身）
            WW = np.zeros(Class_NUM)
            for i_n in range(1, K + 1):
                neighbor_class = knn_idx[i_n] // Class_Train_NUM  # 0-based 类别
                WW[neighbor_class] += 1

            # 计算隶属度
            for i in range(Class_NUM):
                if i == s:
                    U[i, n] = 0.51 + (WW[i] / K) * 0.49
                else:
                    U[i, n] = (WW[i] / K) * 0.49

    # =========================================================
    # 步骤 2: 对测试样本进行模糊 KNN 分类
    # =========================================================
    Miss_NUM = 0
    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            test_sample = Test_DAT[:, j, i]  # shape (DIM,)

            # 与所有训练样本的平方欧氏距离
            diff = Train_SET - test_sample[:, np.newaxis]
            sq_dist = np.einsum('ij,ij->j', diff, diff)

            # 找 K 个最近邻
            _, knn_idx2 = _k_min(sq_dist, K)

            # 计算每类的模糊隶属度 V
            V = np.zeros(Class_NUM)
            for g in range(Class_NUM):
                num = 0.0
                den = 0.0
                for s_k in range(K):
                    idx_k = knn_idx2[s_k]
                    d = sq_dist[idx_k] + np.finfo(float).eps
                    num += U[g, idx_k] / d
                    den += 1.0 / d
                V[g] = num / den

            predict_idx = int(np.argmax(V))  # 0-based
            Predict_Labels[i, j] = predict_idx

            if predict_idx != i:
                Miss_NUM += 1

    return Miss_NUM, Predict_Labels
