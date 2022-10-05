import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# 隨機產生 10 組 2 features 的資料 500 筆 (dy 即原始目標值 label 0~9)
dx, dy = make_blobs(n_samples=500, n_features=2, centers=10, random_state=42)
# 用 KMeans 在資料中找出 5 個分組
kmeans = KMeans(n_clusters=5)
kmeans.fit(dx)
# 預測新的目標值 label
new_dy = kmeans.predict(dx)
plt.rcParams['font.size'] = 14
plt.figure(figsize=(16, 8))
# 以不同顏色畫出原始的 10 群資料
plt.subplot(121)
plt.title('Original data (10 groups)')
plt.scatter(dx.T[0], dx.T[1], c=dy, cmap=plt.cm.Set1)
# 根據重新分成的 5 組來畫出資料
plt.subplot(122)
plt.title('KMeans=5 groups')
plt.scatter(dx.T[0], dx.T[1], c=new_dy, cmap=plt.cm.Set1)
# 顯示圖表
plt.tight_layout()
plt.show()