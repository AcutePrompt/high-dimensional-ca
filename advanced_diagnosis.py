import os
import glob
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score
from sklearn.linear_model import Ridge
from collections import Counter

def find_last_two_snapshots(output_dir="output"):
    files = glob.glob(os.path.join(output_dir, "state_*.npy"))
    if len(files) < 2:
        raise RuntimeError("Need at least two state snapshots in output/")
    # extract timestep correctly
    def ts(fp):
        m = re.search(r"state_(\d+)\.npy$", os.path.basename(fp))
        return int(m.group(1)) if m else -1
    files_sorted = sorted(files, key=ts)
    return files_sorted[-2], files_sorted[-1]

def advanced_diagnosis(output_dir="output"):
    prev_fp, final_fp = find_last_two_snapshots(output_dir)
    t_prev = int(re.search(r"state_(\d+)\.npy$", os.path.basename(prev_fp)).group(1))
    t_final= int(re.search(r"state_(\d+)\.npy$", os.path.basename(final_fp)).group(1))
    print(f"Using snapshots t={t_prev} and t={t_final}")

    X_prev = np.load(prev_fp)
    X_final= np.load(final_fp)

    # 1) Cluster final into 2
    km = KMeans(n_clusters=2, random_state=0).fit(X_final)
    labels = km.labels_
    sizes = Counter(labels)
    sil = silhouette_score(X_final, labels)
    print("Cluster sizes:", sizes)
    print(f"Silhouette score: {sil:.3f}")

    # 2) Centroids (first 10 dims)
    for c in [0,1]:
        cent = X_final[labels==c].mean(axis=0)
        print(f"Cluster {c} centroid dims 0–9:", np.round(cent[:10],3))

    # 3) Ridge regression for cluster 1
    idx1 = np.where(labels==1)[0]
    if len(idx1):
        Xp = X_prev[idx1]
        Xn = X_final[idx1]
        model = Ridge(alpha=1.0)
        model.fit(Xp, Xn)
        preds = model.predict(Xp)
        scores = [r2_score(Xn[:,d], preds[:,d]) for d in range(Xn.shape[1])]
        print(f"Cluster 1 mean R²: {np.mean(scores):.3f}")
    else:
        print("No points in cluster 1 to regress on.")

if __name__=="__main__":
    advanced_diagnosis("output")
