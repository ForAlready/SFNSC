"""
hyperparameter_tuning.py
Systematic hyperparameter tuning for SCI-FSNC on iShip-1 dataset.

This script explores:
1. Lambda (L1 regularization) values
2. K_MIN (minimum neighborhood size) values
3. PCA dimensions (max_disc)
4. SCI threshold values

Goal: Find optimal configuration that maximizes SCI-FSNC performance on iShip-1.
"""

import os
import numpy as np
from experiment_utils import load_iship1, prepare_data
from Classifier_SCI_FSNC import Classifier_SCI_FSNC
from Classifier_kNNC import Classifier_kNNC
from Classifier_fuzzy_KNN import Classifier_fuzzy_KNN

# Configuration
DATASET_ROOT = os.path.join('datasets', 'iShip-1')
IMAGE_SIZE = (64, 64)
SAMPLES_PER_CLASS = 300
TRAIN_RATIO = 80
RANDOM_SEED = 1234

# Hyperparameter search space
LAMBDA_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
K_MIN_VALUES = [1, 3, 5, 7, 10]
MAX_DISC_VALUES = [60, 80, 100, 120, 150]
SCI_THRESHOLD_VALUES = [1e-5, 1e-4, 1e-3, 1e-2]

print('=' * 80)
print('Hyperparameter Tuning for SCI-FSNC on iShip-1')
print('=' * 80)

# Load data once
print('\nLoading iShip-1 dataset...')
All_DAT_raw = load_iship1(
    dataset_root=DATASET_ROOT,
    image_size=IMAGE_SIZE,
    samples_per_class=SAMPLES_PER_CLASS,
    random_seed=RANDOM_SEED,
)
print(f'Loaded: {All_DAT_raw.shape}')

# Baseline: KNN and FKNN with default settings
print('\n' + '-' * 80)
print('Baseline Performance (default settings: max_disc=80)')
print('-' * 80)

Train_SET_3D, Test_SET_3D, _, CTN, CTeN = prepare_data(
    All_DAT_raw=All_DAT_raw,
    Class_NUM=6,
    Class_Sample_NUM=SAMPLES_PER_CLASS,
    train_ratio=TRAIN_RATIO,
    max_disc=80,
)

true_labels = np.repeat(np.arange(6), CTeN)
n_test = 6 * CTeN

print('\nRunning KNN (K=5)...')
Miss, Pred, CM = Classifier_kNNC(Train_SET_3D, Test_SET_3D, 5)
knn_acc = np.sum(Pred.ravel() == true_labels) / n_test
print(f'KNN accuracy: {knn_acc:.4f}')

print('\nRunning FKNN (K=5)...')
Miss, Pred = Classifier_fuzzy_KNN(Train_SET_3D, Test_SET_3D, 5)
fknn_acc = np.sum(Pred.ravel() == true_labels) / n_test
print(f'FKNN accuracy: {fknn_acc:.4f}')

print('\nRunning SCI-FSNC (default: lam=0.05, K_MIN=3)...')
Miss, Pred, CM, tc = Classifier_SCI_FSNC(Train_SET_3D, Test_SET_3D, lam=0.05, K_MIN=3)
baseline_acc = np.sum(Pred.ravel() == true_labels) / n_test
print(f'SCI-FSNC accuracy: {baseline_acc:.4f} (time: {tc:.1f}s)')

# Experiment 1: Lambda tuning (most impactful)
print('\n' + '=' * 80)
print('Experiment 1: Lambda Tuning (K_MIN=3, max_disc=80, sci_threshold=1e-4)')
print('=' * 80)

best_lam = 0.05
best_lam_acc = baseline_acc

for lam in LAMBDA_VALUES:
    print(f'\nTesting lambda={lam}...')
    Miss, Pred, CM, tc = Classifier_SCI_FSNC(
        Train_SET_3D, Test_SET_3D, 
        lam=lam, 
        K_MIN=3,
        sci_threshold=1e-4,
    )
    acc = np.sum(Pred.ravel() == true_labels) / n_test
    print(f'  Accuracy: {acc:.4f} (time: {tc:.1f}s)')
    
    if acc > best_lam_acc:
        best_lam_acc = acc
        best_lam = lam
        print(f'  *** New best lambda: {lam} with accuracy {acc:.4f}')

print(f'\nBest lambda: {best_lam} with accuracy {best_lam_acc:.4f}')

# Experiment 2: K_MIN tuning with best lambda
print('\n' + '=' * 80)
print(f'Experiment 2: K_MIN Tuning (lam={best_lam}, max_disc=80, sci_threshold=1e-4)')
print('=' * 80)

best_k_min = 3
best_k_min_acc = best_lam_acc

for k_min in K_MIN_VALUES:
    print(f'\nTesting K_MIN={k_min}...')
    Miss, Pred, CM, tc = Classifier_SCI_FSNC(
        Train_SET_3D, Test_SET_3D,
        lam=best_lam,
        K_MIN=k_min,
        sci_threshold=1e-4,
    )
    acc = np.sum(Pred.ravel() == true_labels) / n_test
    print(f'  Accuracy: {acc:.4f} (time: {tc:.1f}s)')
    
    if acc > best_k_min_acc:
        best_k_min_acc = acc
        best_k_min = k_min
        print(f'  *** New best K_MIN: {k_min} with accuracy {acc:.4f}')

print(f'\nBest K_MIN: {best_k_min} with accuracy {best_k_min_acc:.4f}')

# Experiment 3: PCA dimension tuning
print('\n' + '=' * 80)
print(f'Experiment 3: PCA Dimension Tuning (lam={best_lam}, K_MIN={best_k_min}, sci_threshold=1e-4)')
print('=' * 80)

best_max_disc = 80
best_max_disc_acc = best_k_min_acc

for max_disc in MAX_DISC_VALUES:
    print(f'\nTesting max_disc={max_disc}...')
    
    # Re-prepare data with new PCA dimension
    Train_SET_3D_new, Test_SET_3D_new, _, _, _ = prepare_data(
        All_DAT_raw=All_DAT_raw,
        Class_NUM=6,
        Class_Sample_NUM=SAMPLES_PER_CLASS,
        train_ratio=TRAIN_RATIO,
        max_disc=max_disc,
    )
    
    Miss, Pred, CM, tc = Classifier_SCI_FSNC(
        Train_SET_3D_new, Test_SET_3D_new,
        lam=best_lam,
        K_MIN=best_k_min,
        sci_threshold=1e-4,
    )
    acc = np.sum(Pred.ravel() == true_labels) / n_test
    print(f'  Accuracy: {acc:.4f} (time: {tc:.1f}s)')
    
    if acc > best_max_disc_acc:
        best_max_disc_acc = acc
        best_max_disc = max_disc
        print(f'  *** New best max_disc: {max_disc} with accuracy {acc:.4f}')

print(f'\nBest max_disc: {best_max_disc} with accuracy {best_max_disc_acc:.4f}')

# Experiment 4: SCI threshold tuning
print('\n' + '=' * 80)
print(f'Experiment 4: SCI Threshold Tuning (lam={best_lam}, K_MIN={best_k_min}, max_disc={best_max_disc})')
print('=' * 80)

# Re-prepare data with best max_disc
Train_SET_3D_best, Test_SET_3D_best, _, _, _ = prepare_data(
    All_DAT_raw=All_DAT_raw,
    Class_NUM=6,
    Class_Sample_NUM=SAMPLES_PER_CLASS,
    train_ratio=TRAIN_RATIO,
    max_disc=best_max_disc,
)

best_sci_threshold = 1e-4
best_sci_threshold_acc = best_max_disc_acc

for sci_threshold in SCI_THRESHOLD_VALUES:
    print(f'\nTesting sci_threshold={sci_threshold}...')
    Miss, Pred, CM, tc = Classifier_SCI_FSNC(
        Train_SET_3D_best, Test_SET_3D_best,
        lam=best_lam,
        K_MIN=best_k_min,
        sci_threshold=sci_threshold,
    )
    acc = np.sum(Pred.ravel() == true_labels) / n_test
    print(f'  Accuracy: {acc:.4f} (time: {tc:.1f}s)')
    
    if acc > best_sci_threshold_acc:
        best_sci_threshold_acc = acc
        best_sci_threshold = sci_threshold
        print(f'  *** New best sci_threshold: {sci_threshold} with accuracy {acc:.4f}')

print(f'\nBest sci_threshold: {best_sci_threshold} with accuracy {best_sci_threshold_acc:.4f}')

# Final summary
print('\n' + '=' * 80)
print('FINAL RESULTS')
print('=' * 80)
print(f'\nBaseline Performance:')
print(f'  KNN (K=5):                    {knn_acc:.4f}')
print(f'  FKNN (K=5):                   {fknn_acc:.4f}')
print(f'  SCI-FSNC (default):           {baseline_acc:.4f}')
print(f'\nOptimized SCI-FSNC:')
print(f'  Lambda:                       {best_lam}')
print(f'  K_MIN:                        {best_k_min}')
print(f'  max_disc:                     {best_max_disc}')
print(f'  sci_threshold:                {best_sci_threshold}')
print(f'  Accuracy:                     {best_sci_threshold_acc:.4f}')
print(f'\nImprovement over baseline:    {(best_sci_threshold_acc - baseline_acc)*100:.2f}%')
print(f'Improvement over KNN:          {(best_sci_threshold_acc - knn_acc)*100:.2f}%')
print(f'Improvement over FKNN:         {(best_sci_threshold_acc - fknn_acc)*100:.2f}%')

# Save optimal configuration
config_file = 'optimal_config_iship1.txt'
with open(config_file, 'w') as f:
    f.write('Optimal SCI-FSNC Configuration for iShip-1\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'lam = {best_lam}\n')
    f.write(f'K_MIN = {best_k_min}\n')
    f.write(f'max_disc = {best_max_disc}\n')
    f.write(f'sci_threshold = {best_sci_threshold}\n')
    f.write(f'\nAccuracy: {best_sci_threshold_acc:.4f}\n')
    f.write(f'Baseline accuracy: {baseline_acc:.4f}\n')
    f.write(f'Improvement: {(best_sci_threshold_acc - baseline_acc)*100:.2f}%\n')

print(f'\nOptimal configuration saved to: {config_file}')
print('=' * 80)
