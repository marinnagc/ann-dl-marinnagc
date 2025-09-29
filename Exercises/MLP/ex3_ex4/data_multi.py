# data_multi.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def gen_multi_clusters(n_total=1500, seed=42):
    per = [2, 3, 4]  # clusters por classe
    n_per_class = [n_total//3]*3
    rng = np.random.default_rng(seed)
    XX, yy = [], []
    rs = seed
    for cls, k in enumerate(per):
        # gerar k blocos e concatenar
        sizes = np.full(k, n_per_class[cls]//k, dtype=int)
        sizes[:(n_per_class[cls] - sizes.sum())] += 1
        parts = []
        for s in sizes:
            Xc, _ = make_classification(n_samples=s, n_features=4,
                                        n_informative=4, n_redundant=0,
                                        n_clusters_per_class=1, class_sep=2.0,
                                        flip_y=0.0, random_state=rs)
            rs += 1
            parts.append(Xc)
        Xcls = np.vstack(parts)
        ycls = np.full(len(Xcls), cls, int)
        XX.append(Xcls); yy.append(ycls)
    X = np.vstack(XX); y = np.concatenate(yy)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte
