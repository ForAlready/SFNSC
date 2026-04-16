import os
import cv2
import numpy as np

path = 'datasets/obj_gray'
CLASS_NAMES = {'01': '浮球', '02': '水上灯塔', '03': '客船', '04': '锥形浮标'}

# 1. 跨类相似度（用第1张图代表每类）
print('=== Inter-class cosine similarity (sample 1 of each class) ===')
cls_vecs = {}
for cls in ['01', '02', '03', '04']:
    fp = os.path.join(path, f'obj_{cls}_1.jpg')
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    cls_vecs[cls] = cv2.resize(img, (64, 64)).ravel().astype(np.float64) / 255.0

clss = list(cls_vecs.keys())
for i in range(len(clss)):
    for j in range(i + 1, len(clss)):
        a, b = cls_vecs[clss[i]], cls_vecs[clss[j]]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f'  C{clss[i]}({CLASS_NAMES[clss[i]]}) vs C{clss[j]}({CLASS_NAMES[clss[j]]}): {sim:.4f}')

# 2. 类内相似度（抽样5张）
print('\n=== Intra-class cosine similarity (samples 1,50,100,150,200) ===')
for cls in ['01', '02', '03', '04']:
    vecs = []
    for t in [1, 50, 100, 150, 200]:
        fp = os.path.join(path, f'obj_{cls}_{t}.jpg')
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        vecs.append(cv2.resize(img, (64, 64)).ravel().astype(np.float64) / 255.0)
    sims = [np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    print(f'  C{cls}({CLASS_NAMES[cls]}): min={min(sims):.4f}  max={max(sims):.4f}  mean={np.mean(sims):.4f}')

# 3. 验证是否存在重复图像（连续帧是否完全相同）
print('\n=== Checking for duplicate frames (consecutive pairs) ===')
for cls in ['01', '02', '03', '04']:
    dups = 0
    for t in range(1, 10):
        fp1 = os.path.join(path, f'obj_{cls}_{t}.jpg')
        fp2 = os.path.join(path, f'obj_{cls}_{t+1}.jpg')
        img1 = cv2.imread(fp1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(fp2, cv2.IMREAD_GRAYSCALE)
        if img1 is not None and img2 is not None:
            img1r = cv2.resize(img1, (64, 64)).ravel().astype(np.float64)
            img2r = cv2.resize(img2, (64, 64)).ravel().astype(np.float64)
            if np.allclose(img1r, img2r, atol=1.0):
                dups += 1
    print(f'  C{cls}: {dups}/9 consecutive pairs are near-identical')
