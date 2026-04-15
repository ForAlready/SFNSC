"""
Classifier_FSNC.py
融合稀疏表示的自适应模糊相似邻居分类器 (FSNC)

阶段 1 (训练): 利用留一法 + NSR 构建模糊隶属度矩阵 U
阶段 2 (测试): 利用 NSR 系数 + U 计算决策隶属度 v，取最大值作为预测标签
"""

import time
import numpy as np
from solve_NSR import solve_NSR


def Classifier_FSNC(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    lam: float,
):
    """
    FSNC 分类器

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (DIM, Class_Train_NUM, Class_NUM)
    Test_DAT  : np.ndarray, shape (DIM, Class_Test_NUM,  Class_NUM)
    lam       : float — 稀疏正则化参数 lambda

    Returns
    -------
    Miss_NUM       : int
    Predict_Labels : np.ndarray, shape (Class_NUM, Class_Test_NUM)  — 0-based 预测标签
    CM             : np.ndarray, shape (Class_NUM, Class_NUM)       — 混淆矩阵
    Time_Cost      : float — 耗时 (秒)
    """
    t_start = time.perf_counter()

    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    Train_NUM = Class_Train_NUM * Class_NUM

    # 重塑为 2D 字典矩阵 (DIM, Train_NUM)，MATLAB column-major
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')

    # =================================================================
    # 阶段 1: 训练 — 构建模糊隶属度矩阵 U (Eq.4)
    # =================================================================
    U = np.zeros((Class_NUM, Train_NUM))

    for s in range(Class_NUM):
        for t in range(Class_Train_NUM):
            n = s * Class_Train_NUM + t  # 0-based 全局索引

            # Leave-one-out: 移除第 n 列
            X_loo = np.delete(Train_SET, n, axis=1)   # (DIM, Train_NUM-1)
            y_loo = Train_SET[:, n]                    # (DIM,)

            # 求非负稀疏系数
            w_loo = solve_NSR(X_loo, y_loo, lam)      # shape (Train_NUM-1,)

            # 在第 n 个位置补 0，恢复长度 Train_NUM
            w_j = np.concatenate([w_loo[:n], [0.0], w_loo[n:]])

            xi_j = np.sum(w_j) + np.finfo(float).eps  # 防止除零

            # 对每个类计算隶属度
            for i in range(Class_NUM):
                start_i = i * Class_Train_NUM
                end_i = start_i + Class_Train_NUM
                xi_ij = np.sum(w_j[start_i:end_i])

                if i == s:
                    U[i, n] = 0.51 + 0.49 * (xi_ij / xi_j)
                else:
                    U[i, n] = 0.49 * (xi_ij / xi_j)

    # =================================================================
    # 阶段 2: 测试 — 基于 U 进行自适应模糊分类 (Eq.5)
    # =================================================================
    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)
    Miss_NUM = 0
    CM = np.zeros((Class_NUM, Class_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            test_sample = Test_DAT[:, j, i]

            # 用全体训练集字典求稀疏系数
            w_y = solve_NSR(Train_SET, test_sample, lam)  # (Train_NUM,)

            # 计算决策隶属度 v (Eq.5)
            sum_w_y = np.sum(w_y) + np.finfo(float).eps
            v = (U @ w_y) / sum_w_y   # shape (Class_NUM,)

            predict_idx = int(np.argmax(v))   # 0-based
            Predict_Labels[i, j] = predict_idx

            # 更新混淆矩阵与错误计数
            CM[i, predict_idx] += 1
            if predict_idx != i:
                Miss_NUM += 1

    Time_Cost = time.perf_counter() - t_start
    return Miss_NUM, Predict_Labels, CM, Time_Cost
