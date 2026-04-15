"""
Classifier_SRC.py
稀疏表示分类器 (Sparse Representation based Classifier, SRC)
对应论文 NEUCOM12784.pdf Section 2.2

基本思路:
  1. 将测试样本 y 用全体训练样本字典稀疏表示: y ≈ Xw
  2. 对每个类 i，利用属于该类的系数子集重建 y，得残差 r_i(y)
  3. 决策规则: 分配到残差最小的类

优化目标: min ||y - Xw||_2^2 + lambda * ||w||_1
使用 ISTA/FISTA 求解（允许系数正负，不施加非负约束）。
"""

import numpy as np
from sklearn.linear_model import Lasso


def _solve_L1(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    求解 L1 正则化最小二乘 (LASSO):
        min_{w} 0.5 * ||y - Xw||_2^2 + lam * ||w||_1
    使用 scikit-learn Lasso。
    """
    lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000, tol=1e-2)
    lasso.fit(X, y)
    return lasso.coef_


def Classifier_SRC(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    lam: float = 0.001,
):
    """
    SRC 分类器

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (DIM, Class_Train_NUM, Class_NUM)
    Test_DAT  : np.ndarray, shape (DIM, Class_Test_NUM,  Class_NUM)
    lam       : float — L1 正则化参数

    Returns
    -------
    Miss_NUM       : int
    Predict_Labels : np.ndarray, shape (Class_NUM, Class_Test_NUM) — 0-based
    CM             : np.ndarray, shape (Class_NUM, Class_NUM)
    """
    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    Train_NUM = Class_Train_NUM * Class_NUM
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')

    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)
    Miss_NUM = 0
    CM = np.zeros((Class_NUM, Class_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            y = Test_DAT[:, j, i]

            # 稀疏表示
            w = _solve_L1(Train_SET, y, lam)   # shape (Train_NUM,)

            # 对每个类计算残差
            residuals = np.zeros(Class_NUM)
            for c in range(Class_NUM):
                # 属于第 c 类的系数索引
                idx_c = slice(c * Class_Train_NUM, (c + 1) * Class_Train_NUM)
                w_c = np.zeros(Train_NUM)
                w_c[idx_c] = w[idx_c]
                reconstructed = Train_SET @ w_c
                residuals[c] = np.linalg.norm(y - reconstructed)

            predict_idx = int(np.argmin(residuals))
            Predict_Labels[i, j] = predict_idx
            CM[i, predict_idx] += 1
            if predict_idx != i:
                Miss_NUM += 1

    return Miss_NUM, Predict_Labels, CM
