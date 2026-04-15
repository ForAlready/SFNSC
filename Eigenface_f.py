"""
Eigenface_f.py
PCA 特征向量提取 (Eigenface 方法)
"""

import numpy as np


def Eigenface_f(Train_DAT: np.ndarray, Disc_NUM: int):
    """
    计算训练数据的 PCA 特征向量 (Eigenfaces)

    Parameters
    ----------
    Train_DAT : np.ndarray, shape (D, N)  — 训练数据矩阵，每列一个样本
    Disc_NUM  : int                        — 需要提取的主成分数量

    Returns
    -------
    disc_set   : np.ndarray, shape (D, Disc_NUM)  — 前 Disc_NUM 个特征向量（列向量）
    disc_value : np.ndarray, shape (Disc_NUM,)    — 对应的特征值（降序）
    """
    # numpy cov 默认对 行 求协方差，MATLAB cov(X') 等价于 np.cov(Train_DAT)
    # 即把每一 行 当作一个变量，每一 列 当作一个观测
    # MATLAB: cov(Train_DAT') — Train_DAT' 的每行是一个样本
    # 等价于 numpy: np.cov(Train_DAT)  — Train_DAT 的每列是一个样本
    C = np.cov(Train_DAT)  # shape (D, D)

    # 求特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # eigh 保证对称矩阵数值稳定

    # 降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 取前 Disc_NUM 个
    Disc_NUM = min(Disc_NUM, eigenvectors.shape[1])
    disc_set = eigenvectors[:, :Disc_NUM]      # shape (D, Disc_NUM)
    disc_value = eigenvalues[:Disc_NUM]        # shape (Disc_NUM,)

    return disc_set, disc_value
