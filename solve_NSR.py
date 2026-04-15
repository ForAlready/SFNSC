import numpy as np
from sklearn.linear_model import Lasso


def solve_NSR(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    非负稀疏表示求解器 (NSR Solver via scikit-learn Lasso)

    Parameters
    ----------
    X     : np.ndarray, shape (D, N)  — 字典矩阵，每列一个原子
    y     : np.ndarray, shape (D,)    — 待表示的样本向量
    lam   : float                     — L1 正则化参数 (lambda)

    Returns
    -------
    w     : np.ndarray, shape (N,)    — 非负稀疏系数向量
    """
    lasso = Lasso(alpha=lam, positive=True, fit_intercept=False, max_iter=10000, tol=1e-2)
    lasso.fit(X, y)
    return lasso.coef_
