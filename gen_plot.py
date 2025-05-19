import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import argparse

def load_snapshots(output_dir):
    state_files = sorted(glob.glob(os.path.join(output_dir, 'state_*.npy')))
    field_files = sorted(glob.glob(os.path.join(output_dir, 'field_*.npy')))
    # group by feature dimension
    groups = {}
    for sf in state_files:
        arr = np.load(sf)
        D = arr.shape[1]
        m = re.match(r'.*state_(\d+)\.npy$', os.path.basename(sf))
        if not m: continue
        t = int(m.group(1))
        groups.setdefault(D, []).append((t, sf))
    if not groups:
        raise RuntimeError('No state snapshots found')
    # choose largest group
    best_D = max(groups, key=lambda d: len(groups[d]))
    infos = sorted(groups[best_D], key=lambda x: x[0])
    times, states = zip(*infos)
    # match fields
    fmap = {int(re.match(r'.*field_(\d+)\.npy$', os.path.basename(ff)).group(1)): ff for ff in field_files if re.match(r'.*field_(\d+)\.npy$', os.path.basename(ff))}
    matched = [(t, sf, fmap[t]) for t, sf in zip(times, states) if t in fmap]
    if not matched:
        raise RuntimeError('No matching field files for state snapshots')
    times, states, fields = zip(*matched)
    return list(times), list(states), list(fields), best_D

def diagnose_cluster(times, state_paths, n_clusters=2):
    # load final snapshot
    Xf = np.load(state_paths[-1])
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(Xf)
    labels = km.labels_
    counts = Counter(labels)
    print(f'K={n_clusters} cluster sizes:', counts)
    # centroids
    for c in range(n_clusters):
        centroid = Xf[labels==c].mean(axis=0)
        print(f'Cluster {c} centroid (first 5 dims):', centroid[:5])
    return labels

def plot_trajectory(times, state_paths, labels, output_dir):
    # fraction of cluster1 over time
    frac = []
    for sf in state_paths:
        X = np.load(sf)
        # assign via KMeans on final? reuse final labels? Here recluster each time
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        l = km.labels_
        frac.append(np.mean(l==1))
    plt.figure()
    plt.plot(times, frac, 'o-')
    plt.xlabel('Time')
    plt.ylabel('Fraction in cluster 1')
    plt.title('Cluster 1 Fraction Over Time')
    plt.tight_layout()
    path = os.path.join(output_dir, 'cluster_fraction.png')
    plt.savefig(path)
    print('Saved', path)

    # PCA of final cluster separation
    Xf = np.load(state_paths[-1])
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xf)
    plt.figure()
    plt.scatter(Z[labels==0,0], Z[labels==0,1], alpha=0.3, label='cluster 0')
    plt.scatter(Z[labels==1,0], Z[labels==1,1], alpha=0.3, label='cluster 1')
    plt.legend()
    plt.title('Final PCA Cluster Separation')
    plt.tight_layout()
    path = os.path.join(output_dir, 'pca_clusters.png')
    plt.savefig(path)
    print('Saved', path)

if __name__=='__main__':
    p = argparse.ArgumentParser(description='Comprehensive cluster diagnostics')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--n_clusters', type=int, default=2)
    args = p.parse_args()

    times, states, fields, D = load_snapshots(args.output_dir)
    print('Analyzing state_dim=', D, 'at times', times)
    # field norm plot
    norms = [np.linalg.norm(np.load(f)) for f in fields]
    plt.figure(); plt.plot(times, norms, 'o-');
    plt.xlabel('Time'); plt.ylabel('||F||'); plt.title('Global Field Norm'); plt.tight_layout();
    f1 = os.path.join(args.output_dir, 'field_norm.png'); plt.savefig(f1); print('Saved', f1)
    # diagnose final cluster structure
    labels = diagnose_cluster(times, states, args.n_clusters)
    # trajectory and PCA cluster plots
    plot_trajectory(times, states, labels, args.output_dir)
