"""
Classifier_SCI_FSNC.py
融合稀疏表示的自适应模糊水面目标分类器 (终极双支路融合版)
核心创新：
1. 支路一：基于 SCI 动态确定最优邻域 K 值的模糊隶属度判定
2. 支路二：保留完整协同特征的稀疏重构误差判定
3. 动态融合：利用 SCI 作为门控因子，将两者完美结合 (严格对应开题报告)
"""

import time
import numpy as np
from solve_NSR import solve_NSR


def _safe_normalize(
    scores: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalizes scores to probability distribution with numerical stability safeguards.
    
    This helper function provides epsilon-based division safety for all normalization
    operations throughout the classifier. It ensures that division by zero is prevented
    and provides a uniform distribution fallback when the sum is near zero.
    
    Args:
        scores: Score vector to normalize (any shape)
        epsilon: Numerical stability constant for division safety (default: 1e-8)
    
    Returns:
        Normalized probability distribution with same shape as input
        - All elements in [0.0, 1.0]
        - Sum equals 1.0 (within numerical precision)
        - Uniform distribution if input sum < epsilon
    
    Algorithm:
        1. Compute sum of scores
        2. If sum < epsilon: return uniform distribution (1/n for each element)
        3. Otherwise: return scores / sum
        4. Optionally clip to [0.0, 1.0] range for safety
    
    Edge cases:
        - Zero sum: Returns uniform distribution
        - Negative values: Preserved (not clipped unless specified)
        - Single element: Returns [1.0]
        - Empty array: Returns empty array
    """
    # Handle empty array edge case
    if scores.size == 0:
        return scores
    
    # Compute sum with epsilon-based safety
    total = np.sum(scores)
    
    # Uniform distribution fallback when sum is near zero
    if total < epsilon:
        # Return uniform distribution: 1/n for each element
        return np.ones_like(scores) / scores.size
    
    # Normalize to probability distribution
    normalized = scores / total
    
    # Apply clipping to probability values for additional safety
    # Ensures all values are in valid [0.0, 1.0] range
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized


def calculate_sci_active(alpha: np.ndarray, threshold: float = 1e-4, epsilon: float = 1e-8) -> float:
    """
    Computes Sparse Concentration Index with robust edge case handling.
    
    Args:
        alpha: Sparse coefficient vector
        threshold: Activity threshold for coefficient detection (default: 1e-4)
        epsilon: Numerical stability constant (default: 1e-8)
    
    Returns:
        SCI value in range [0.0, 1.0]
        - 1.0 indicates maximum concentration (few dominant coefficients)
        - 0.0 indicates uniform distribution (dispersed coefficients)
    
    Edge cases:
        - n <= 1: Returns 1.0 (maximum concentration)
        - l2_norm == 0: Returns 0.0 (uniform/zero coefficients)
    """
    alpha = np.abs(alpha)
    active_alpha = alpha[alpha > threshold]
    n = len(active_alpha)
    
    # Edge case: empty or single-element active set
    if n <= 1:
        return 1.0 
    
    l1_norm = np.sum(active_alpha)
    l2_norm = np.linalg.norm(active_alpha, 2)
    
    # Edge case: zero L2 norm (all coefficients identical or zero)
    if l2_norm == 0:
        return 0.0 
    
    # SCI formula with epsilon-based numerical stability
    sci = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1 + epsilon)
    
    # Clip to valid probability range
    return np.clip(sci, 0.0, 1.0)


def _compute_adaptive_K(
    w_y: np.ndarray,
    sci_val: float,
    K_MIN: int = 3,
    threshold: float = 1e-5
) -> int:
    """
    Determines optimal neighborhood size based on SCI value.
    
    Strategy:
    - High SCI (concentrated coefficients) → Small K (trust few strong neighbors)
    - Low SCI (dispersed coefficients) → Large K (aggregate more information)
    
    Args:
        w_y: Sparse coefficient vector
        sci_val: Sparse Concentration Index value in [0.0, 1.0]
        K_MIN: Minimum neighborhood size (default: 3)
        threshold: Activity threshold for counting active coefficients (default: 1e-5)
    
    Returns:
        K_dynamic: Adaptive neighborhood size in range [K_MIN, K_MAX]
    
    Formula:
        K_dynamic = round(K_MAX - sci_val * (K_MAX - K_MIN))
        where K_MAX = count of active coefficients
    
    Edge cases:
        - If K_MAX <= K_MIN: returns max(1, active_count)
        - If active_count == 0: returns K_MIN
    """
    # Count active coefficients using configurable threshold
    active_count = np.sum(w_y > threshold)
    
    # Edge case: no active coefficients
    if active_count == 0:
        return K_MIN
    
    K_MAX = active_count
    
    # Edge case: K_MAX <= K_MIN
    if K_MAX <= K_MIN:
        return max(1, active_count)
    
    # Implement inverse SCI mapping formula
    # High SCI → K close to K_MIN (concentrated, trust few neighbors)
    # Low SCI → K close to K_MAX (dispersed, aggregate more neighbors)
    K_dynamic = int(np.round(K_MAX - sci_val * (K_MAX - K_MIN)))
    
    # Enforce bounds: clip K_dynamic to [K_MIN, K_MAX]
    K_dynamic = int(np.clip(K_dynamic, K_MIN, K_MAX))
    
    return K_dynamic


def _compute_fuzzy_voting_score(
    U: np.ndarray,
    w_y: np.ndarray,
    K_dynamic: int,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Computes fuzzy membership-weighted voting score.
    
    This function implements Branch_One of the SCI-FSNC classifier, which uses
    adaptive K-nearest neighbor voting weighted by fuzzy membership values.
    
    Args:
        U: Fuzzy membership matrix with shape (Class_NUM, Train_NUM)
           U[i, j] represents the fuzzy membership of training sample j to class i
        w_y: Sparse coefficient vector with shape (Train_NUM,)
             Non-negative coefficients from NSR solver
        K_dynamic: Adaptive neighborhood size (number of top coefficients to use)
        epsilon: Numerical stability constant for division safety (default: 1e-8)
    
    Returns:
        V: Fuzzy voting score vector with shape (Class_NUM,)
           Normalized probability distribution summing to 1.0
    
    Algorithm:
        1. Truncate w_y to top-K_dynamic coefficients using np.argsort
        2. Compute weighted fuzzy membership: V_raw = U @ w_trunc
        3. Normalize by coefficient sum: V = V_raw / sum(w_trunc)
        4. Normalize by score sum: V = V / sum(V)
        5. Add epsilon-based safety for zero denominators
    
    Edge cases:
        - K_dynamic == 0: Returns uniform distribution
        - All coefficients zero: Returns uniform distribution
        - Zero membership sum: Returns uniform distribution
    """
    Class_NUM = U.shape[0]
    
    # Edge case: K_dynamic is 0 or negative
    if K_dynamic <= 0:
        return np.ones(Class_NUM) / Class_NUM
    
    # Implement top-K coefficient truncation using np.argsort
    w_trunc = np.zeros_like(w_y)
    if K_dynamic > 0:
        # Find indices of top-K_dynamic coefficients
        top_k_indices = np.argsort(w_y)[-K_dynamic:]
        w_trunc[top_k_indices] = w_y[top_k_indices]
    
    # Compute weighted fuzzy membership: V_raw = U @ w_trunc
    V_raw = U @ w_trunc
    
    # Implement double normalization with epsilon-based safety using _safe_normalize
    # First normalization: by coefficient sum
    sum_w_trunc = np.sum(w_trunc) + epsilon
    V = V_raw / sum_w_trunc
    
    # Second normalization: by score sum (convert to probability distribution)
    # Use _safe_normalize helper for numerical stability
    V = _safe_normalize(V, epsilon)
    
    return V


def _compute_reconstruction_score(
    Train_SET: np.ndarray,
    test_sample: np.ndarray,
    w_y: np.ndarray,
    Class_NUM: int,
    Class_Train_NUM: int,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Computes class-specific reconstruction residual scores.
    
    This function implements Branch_Two of the SCI-FSNC classifier, which uses
    sparse reconstruction residuals to compute class scores. Unlike Branch_One,
    this branch uses the FULL untruncated sparse coefficient vector to preserve
    the mathematical optimality of the L1 solution.
    
    Args:
        Train_SET: Training data matrix with shape (DIM, Train_NUM)
                   Reshaped from 3D to 2D using Fortran column-major order
        test_sample: Test sample vector with shape (DIM,)
        w_y: Complete untruncated sparse coefficient vector with shape (Train_NUM,)
             Non-negative coefficients from NSR solver
        Class_NUM: Number of classes
        Class_Train_NUM: Number of training samples per class
        epsilon: Numerical stability constant for division safety (default: 1e-6)
    
    Returns:
        S: Reconstruction score vector with shape (Class_NUM,)
           Normalized probability distribution summing to 1.0
    
    Algorithm:
        1. For each class c:
           a. Extract class-specific training data using slice indexing:
              X_c = Train_SET[:, c*Class_Train_NUM : (c+1)*Class_Train_NUM]
           b. Extract class-specific coefficients:
              w_c = w_y[c*Class_Train_NUM : (c+1)*Class_Train_NUM]
           c. Compute per-class reconstruction:
              y_recon_c = X_c @ w_c
           d. Compute L2 reconstruction error:
              error_c = ||test_sample - y_recon_c||_2
        2. Convert errors to scores using inverse transformation:
           score_c = 1.0 / (error_c + epsilon)
        3. Normalize scores to probability distribution:
           S = scores / sum(scores)
    
    Rationale for using full w_y (no truncation):
        - Truncation would violate optimality of Lasso solution
        - Small coefficients still contribute to accurate reconstruction
        - Branch_One handles noise filtering via K-truncation
    
    Edge cases:
        - Perfect reconstruction (zero error): score = 1/epsilon (dominant)
        - Equal errors across classes: uniform distribution
        - Zero score sum: uniform distribution fallback
    """
    # Initialize inverse error scores
    E_inv = np.zeros(Class_NUM)
    
    # Compute per-class reconstruction errors
    for c in range(Class_NUM):
        # Implement class-specific coefficient extraction using slice indexing
        start_idx = c * Class_Train_NUM
        end_idx = start_idx + Class_Train_NUM
        
        # Extract class-specific training data
        X_c = Train_SET[:, start_idx:end_idx]
        
        # Extract class-specific coefficients from full w_y (no truncation)
        w_c = w_y[start_idx:end_idx]
        
        # Compute per-class reconstruction: y_recon_c = X_c @ w_c
        y_recon_c = X_c @ w_c
        
        # Compute L2 reconstruction error
        error_c = np.linalg.norm(test_sample - y_recon_c, 2)
        
        # Convert errors to scores using inverse transformation
        # Lower error → higher score
        E_inv[c] = 1.0 / (error_c + epsilon)
    
    # Normalize scores to probability distribution using _safe_normalize
    # This handles edge cases like zero score sum with uniform distribution fallback
    S = _safe_normalize(E_inv, epsilon)
    
    return S


def _build_fuzzy_membership_matrix(
    Train_DAT: np.ndarray,
    lam: float,
    epsilon: float = 1e-8,
    verbose: bool = False
) -> np.ndarray:
    """
    Constructs fuzzy membership matrix via leave-one-out NSR.
    
    This function implements the training phase of SCI-FSNC, computing fuzzy
    membership values for all training samples using a leave-one-out approach.
    Each training sample is removed, NSR is solved, and membership values are
    assigned based on class-specific coefficient sums.
    
    Args:
        Train_DAT: Training data array with shape (DIM, Class_Train_NUM, Class_NUM)
                   3D array where each "slice" along axis 2 contains samples from one class
        lam: L1 regularization parameter for NSR solver
        epsilon: Numerical stability constant for division safety (default: 1e-8)
        verbose: Enable progress reporting (default: False)
    
    Returns:
        U: Fuzzy membership matrix with shape (Class_NUM, Train_NUM)
           U[i, j] represents the fuzzy membership of training sample j to class i
           Each column sums to approximately 1.0
           Same-class memberships are > 0.5, different-class memberships are < 0.5
    
    Algorithm:
        1. Reshape Train_DAT to 2D format (DIM, Train_NUM) using Fortran order
        2. Initialize U = zeros(Class_NUM, Train_NUM)
        3. For each training sample n (leave-one-out loop):
           a. Remove sample n from Train_SET: X_loo = Train_SET[:, ≠n]
           b. Solve NSR: w_loo = solve_NSR(X_loo, Train_SET[:, n], lam)
           c. Reconstruct full coefficient vector w_j by inserting 0 at position n
           d. Compute total coefficient sum: xi_j = sum(w_j) + epsilon
           e. For each class i:
              - Compute class-specific coefficient sum: xi_ij = sum(w_j[class_i_indices])
              - If i == true_class: U[i, n] = 0.51 + 0.49 * (xi_ij / xi_j)
              - Else: U[i, n] = 0.49 * (xi_ij / xi_j)
        4. Return U matrix
    
    Membership Assignment Strategy:
        - Same-class samples: 0.51 + 0.49 * ratio
          Ensures membership > 0.5, with ratio providing fine-grained discrimination
        - Different-class samples: 0.49 * ratio
          Ensures membership < 0.5, allowing for fuzzy overlap between classes
        - The 0.51/0.49 split ensures same-class dominance while preserving fuzzy properties
    
    Computational Complexity:
        - Time: O(Train_NUM²) due to leave-one-out loop with NSR solving
        - Space: O(Class_NUM × Train_NUM) for U matrix
    
    Edge cases:
        - Zero coefficient sum: epsilon prevents division by zero
        - Single training sample per class: still computes valid memberships
        - NSR solver failure: handled by solve_NSR (returns zero coefficients)
    """
    # Extract dimensions from 3D training data
    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    Train_NUM = Class_Train_NUM * Class_NUM
    
    # Reshape Train_DAT to 2D format (DIM, Train_NUM) using Fortran column-major order
    # This maintains MATLAB-style column ordering where samples from same class are contiguous
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')
    
    # Initialize fuzzy membership matrix U with shape (Class_NUM, Train_NUM)
    U = np.zeros((Class_NUM, Train_NUM))
    
    # Implement leave-one-out loop over all training samples
    for s in range(Class_NUM):
        for t in range(Class_Train_NUM):
            # Compute linear index n for current sample
            n = s * Class_Train_NUM + t
            
            # Progress reporting when verbose mode is enabled
            if verbose and (n % 100 == 0 or n == Train_NUM - 1):
                print(f"Building fuzzy membership matrix: {n+1}/{Train_NUM} samples processed")
            
            # For each sample: remove from Train_SET
            # X_loo excludes column n (leave-one-out)
            X_loo = np.delete(Train_SET, n, axis=1)
            
            # y_loo is the sample being left out
            y_loo = Train_SET[:, n]
            
            # Solve NSR to obtain sparse coefficients w_loo
            # w_loo has shape (Train_NUM - 1,) since we removed one sample
            # Wrap in try-except for graceful degradation on solver failure
            try:
                w_loo = solve_NSR(X_loo, y_loo, lam)
            except Exception as e:
                # Log warning when solver fails (if verbose=True)
                if verbose:
                    print(f"Warning: NSR solver failed for training sample {n} (class={s}, idx={t}): {e}")
                    print(f"         Using zero coefficient fallback for this sample")
                # Use zero coefficient fallback on solver failure
                # This ensures graceful degradation without crashing
                w_loo = np.zeros(X_loo.shape[1])
            
            # Reconstruct full coefficient vector w_j by inserting 0 at position n
            # This maintains alignment with original Train_SET indexing
            w_j = np.concatenate([w_loo[:n], [0.0], w_loo[n:]])
            
            # Compute class-specific coefficient sums (xi_ij)
            # Add epsilon to denominator for numerical stability (prevents division by zero)
            xi_j = np.sum(w_j) + epsilon
            
            # Assign membership values for all classes
            for i in range(Class_NUM):
                # Compute start and end indices for class i samples
                start_i = i * Class_Train_NUM
                end_i = start_i + Class_Train_NUM
                
                # Compute class-specific coefficient sum xi_ij
                xi_ij = np.sum(w_j[start_i:end_i])
                
                # Assign membership values based on whether i is the true class
                if i == s:
                    # Same-class membership: 0.51 + 0.49 * ratio
                    # Ensures membership > 0.5 for true class
                    U[i, n] = 0.51 + 0.49 * (xi_ij / xi_j)
                else:
                    # Different-class membership: 0.49 * ratio
                    # Ensures membership < 0.5 for other classes
                    U[i, n] = 0.49 * (xi_ij / xi_j)
    
    if verbose:
        print(f"Fuzzy membership matrix construction complete: U shape = {U.shape}")
    
    return U


def _validate_inputs(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    lam: float,
    K_MIN: int = 3
) -> None:
    """
    Validates all input parameters before processing.
    
    Args:
        Train_DAT: Training data array
        Test_DAT: Test data array
        lam: L1 regularization parameter
        K_MIN: Minimum neighborhood size
    
    Raises:
        ValueError: If any validation check fails
    """
    # Shape validation - must be 3D arrays
    if Train_DAT.ndim != 3:
        raise ValueError(f"Train_DAT must be 3D array, got shape {Train_DAT.shape} with {Train_DAT.ndim} dimensions")
    
    if Test_DAT.ndim != 3:
        raise ValueError(f"Test_DAT must be 3D array, got shape {Test_DAT.shape} with {Test_DAT.ndim} dimensions")
    
    # Dimension consistency checks
    if Train_DAT.shape[0] != Test_DAT.shape[0]:
        raise ValueError(
            f"Feature dimension (DIM) mismatch: "
            f"Train_DAT has {Train_DAT.shape[0]} features, "
            f"Test_DAT has {Test_DAT.shape[0]} features"
        )
    
    if Train_DAT.shape[2] != Test_DAT.shape[2]:
        raise ValueError(
            f"Class count (Class_NUM) mismatch: "
            f"Train_DAT has {Train_DAT.shape[2]} classes, "
            f"Test_DAT has {Test_DAT.shape[2]} classes"
        )
    
    # Hyperparameter validation
    if lam <= 0:
        raise ValueError(f"Lambda (lam) must be positive, got {lam}")
    
    if K_MIN < 1:
        raise ValueError(f"K_MIN must be at least 1, got {K_MIN}")
    
    # Data quality checks - NaN detection
    if np.any(np.isnan(Train_DAT)):
        raise ValueError("Train_DAT contains NaN values")
    
    if np.any(np.isnan(Test_DAT)):
        raise ValueError("Test_DAT contains NaN values")
    
    # Data quality checks - Inf detection
    if np.any(np.isinf(Train_DAT)):
        raise ValueError("Train_DAT contains infinite values")
    
    if np.any(np.isinf(Test_DAT)):
        raise ValueError("Test_DAT contains infinite values")


def Classifier_SCI_FSNC(
    Train_DAT: np.ndarray,
    Test_DAT: np.ndarray,
    lam: float,
    K_MIN: int = 3,
    sci_threshold: float = 1e-4,
    epsilon: float = 1e-8,
    verbose: bool = False
):
    """
    SCI-FSNC (Sparse Concentration Index - Fuzzy Sparse Neighbor Classifier)
    
    A dual-branch classifier that dynamically fuses fuzzy membership voting with
    sparse reconstruction residuals, weighted by a Sparse Concentration Index (SCI).
    
    Args:
        Train_DAT: Training data array with shape (DIM, Class_Train_NUM, Class_NUM)
                   3D array where each "slice" along axis 2 contains samples from one class
        Test_DAT: Test data array with shape (DIM, Class_Test_NUM, Class_NUM)
                  3D array with same structure as Train_DAT
        lam: L1 regularization parameter for sparse representation (Lambda)
             Controls sparsity level in NSR solver (default: 0.05 in experiments)
             Must be positive (lam > 0)
        K_MIN: Minimum neighborhood size for adaptive K selection (default: 3)
               Lower bound for K_dynamic in fuzzy voting branch
               Must be at least 1 (K_MIN >= 1)
        sci_threshold: Activity threshold for coefficient detection (default: 1e-4)
                       Coefficients with |alpha| > sci_threshold are considered active
                       Used in SCI calculation and adaptive K selection
        epsilon: Numerical stability constant for division safety (default: 1e-8)
                 Added to all denominators to prevent division by zero
                 Used throughout normalization and score computation
        verbose: Enable diagnostic outputs (default: False)
                 When True, prints per-sample SCI values, K_dynamic, branch scores,
                 and final fused scores during testing phase
    
    Returns:
        tuple: (Miss_NUM, Predict_Labels, CM, Time_Cost)
            Miss_NUM: int
                Total count of misclassified test samples
            Predict_Labels: np.ndarray
                Predicted class labels with shape (Class_NUM, Class_Test_NUM)
                Contains 0-based integer class indices
            CM: np.ndarray
                Confusion matrix with shape (Class_NUM, Class_NUM)
                CM[i, j] represents samples from true class i predicted as class j
            Time_Cost: float
                Total execution time in seconds (measured using perf_counter)
    
    Algorithm Overview:
        Phase 1 - Training:
            1. Construct fuzzy membership matrix U via leave-one-out NSR
            2. U[i, j] represents fuzzy membership of training sample j to class i
        
        Phase 2 - Testing (for each test sample):
            1. Solve NSR to obtain sparse coefficients w_y
            2. Compute SCI value from w_y
            3. Branch_One: Fuzzy voting with adaptive K
               - Determine K_dynamic based on SCI
               - Truncate w_y to top-K coefficients
               - Compute V = normalize(U @ w_trunc)
            4. Branch_Two: Reconstruction residual
               - Use full w_y (no truncation)
               - Compute per-class reconstruction errors
               - Compute S = normalize(1 / (errors + epsilon))
            5. Fusion: Final_Score = (1 - SCI) * V + SCI * S
            6. Predict: argmax(Final_Score)
    
    Raises:
        ValueError: If input validation fails (invalid shapes, dimensions, hyperparameters, or data quality)
    
    Example:
        >>> # Basic usage with required parameters only
        >>> Miss_NUM, Predict_Labels, CM, Time_Cost = Classifier_SCI_FSNC(
        ...     Train_DAT, Test_DAT, lam=0.05
        ... )
        
        >>> # Advanced usage with custom hyperparameters
        >>> Miss_NUM, Predict_Labels, CM, Time_Cost = Classifier_SCI_FSNC(
        ...     Train_DAT, Test_DAT, 
        ...     lam=0.05, 
        ...     K_MIN=5, 
        ...     sci_threshold=1e-3,
        ...     epsilon=1e-8,
        ...     verbose=True
        ... )
    
    References:
        - Requirements: 5.1, 5.2, 5.3, 5.4, 9.5
        - Design: See design.md for detailed architecture and component descriptions
    """
    # Validate inputs at function start
    _validate_inputs(Train_DAT, Test_DAT, lam, K_MIN)
    
    t_start = time.perf_counter()

    DIM, Class_Train_NUM, Class_NUM = Train_DAT.shape
    _, Class_Test_NUM, _ = Test_DAT.shape

    Train_NUM = Class_Train_NUM * Class_NUM
    Train_SET = Train_DAT.reshape(DIM, Train_NUM, order='F')

    # =================================================================
    # 阶段 1: 训练 — 构建模糊隶属度矩阵 U
    # =================================================================
    # Replace inline membership construction with helper function call
    U = _build_fuzzy_membership_matrix(Train_DAT, lam, epsilon=epsilon, verbose=verbose)

    # =================================================================
    # 阶段 2: 测试 — 【核心创新】双支路联合决策
    # =================================================================
    Predict_Labels = np.zeros((Class_NUM, Class_Test_NUM), dtype=int)
    Miss_NUM = 0
    CM = np.zeros((Class_NUM, Class_NUM), dtype=int)

    for i in range(Class_NUM):
        for j in range(Class_Test_NUM):
            test_sample = Test_DAT[:, j, i]

            # 1. 求解全局稀疏重构系数 (完整保留最优解)
            # Wrap in try-except for graceful degradation on solver failure
            try:
                w_y = solve_NSR(Train_SET, test_sample, lam)
            except Exception as e:
                # Log warning when solver fails (if verbose=True)
                if verbose:
                    print(f"Warning: NSR solver failed for test sample (class={i}, idx={j}): {e}")
                    print(f"         Using zero coefficient fallback for this sample")
                # Use zero coefficient fallback on solver failure
                # This ensures graceful degradation without crashing
                w_y = np.zeros(Train_SET.shape[1])

            # 2. 计算浓度指数 SCI
            sci_val = calculate_sci_active(w_y, threshold=sci_threshold, epsilon=epsilon)

            # ---------------------------------------------------------
            # 【支路一】：基于自适应 K 值的模糊隶属度得分 (V)
            # ---------------------------------------------------------
            # Replace inline K selection with helper function call
            K_dynamic = _compute_adaptive_K(w_y, sci_val, K_MIN=K_MIN, threshold=sci_threshold)

            # Replace inline fuzzy voting with helper function call
            V = _compute_fuzzy_voting_score(U, w_y, K_dynamic, epsilon=epsilon)

            # ---------------------------------------------------------
            # 【支路二】：基于完整权重的稀疏重构残差得分 (S)
            # ---------------------------------------------------------
            # Replace inline reconstruction with helper function call
            S = _compute_reconstruction_score(
                Train_SET, 
                test_sample, 
                w_y, 
                Class_NUM, 
                Class_Train_NUM, 
                epsilon=epsilon
            )

            # ---------------------------------------------------------
            # 【终极融合】：利用 SCI 动态分配支路权重
            # SCI 越高，越信任重构残差；SCI 越低，越信任模糊投票
            # ---------------------------------------------------------
            Final_Score = (1.0 - sci_val) * V + sci_val * S

            predict_idx = int(np.argmax(Final_Score))
            Predict_Labels[i, j] = predict_idx
            
            # Verbose diagnostic outputs when enabled
            # Format as readable JSON-like structure
            if verbose:
                # Count active coefficients for additional context
                active_count = np.sum(w_y > sci_threshold)
                
                print(f"\n{'='*60}")
                print(f"Sample Index: (class={i}, sample={j})")
                print(f"{'='*60}")
                print(f"Sparse Representation:")
                print(f"  - Active coefficients: {active_count}")
                print(f"  - SCI value: {sci_val:.6f}")
                print(f"  - K_dynamic: {K_dynamic}")
                print(f"\nBranch Scores (before fusion):")
                print(f"  - Branch_One (Fuzzy Voting):")
                print(f"      {np.array2string(V, precision=6, separator=', ', suppress_small=True)}")
                print(f"  - Branch_Two (Reconstruction):")
                print(f"      {np.array2string(S, precision=6, separator=', ', suppress_small=True)}")
                print(f"\nFusion:")
                print(f"  - Weight for Branch_One: {1.0 - sci_val:.6f}")
                print(f"  - Weight for Branch_Two: {sci_val:.6f}")
                print(f"  - Final_Score:")
                print(f"      {np.array2string(Final_Score, precision=6, separator=', ', suppress_small=True)}")
                print(f"\nPrediction:")
                print(f"  - Predicted class: {predict_idx}")
                print(f"  - True class: {i}")
                print(f"  - Correct: {'✓' if predict_idx == i else '✗'}")
                print(f"{'='*60}\n")

            CM[i, predict_idx] += 1
            if predict_idx != i:
                Miss_NUM += 1

    Time_Cost = time.perf_counter() - t_start
    return Miss_NUM, Predict_Labels, CM, Time_Cost
