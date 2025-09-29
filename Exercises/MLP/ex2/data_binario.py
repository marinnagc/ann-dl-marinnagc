# data_binario.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def gen_binary_clusters(n_total=1000, seed=42):
    rng = np.random.default_rng(seed)
    # Classe 0 com 1 cluster
    X0, y0 = make_classification(n_samples=n_total//2, n_features=2,
                                 n_informative=2, n_redundant=0,
                                 n_clusters_per_class=1, class_sep=1.5,
                                 flip_y=0.01, random_state=seed)
    y0[:] = 0
    # Classe 1 com 2 clusters -> geramos duas metades e juntamos
    X1a, y1a = make_classification(n_samples=n_total//4, n_features=2,
                                   n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, class_sep=1.6,
                                   flip_y=0.01, random_state=seed+1)
    X1b, y1b = make_classification(n_samples=n_total//4, n_features=2,
                                   n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, class_sep=1.6,
                                   flip_y=0.01, random_state=seed+2)
    X = np.vstack([X0, X1a, X1b])
    y = np.concatenate([np.zeros(len(X0), int),
                        np.ones(len(X1a), int),
                        np.ones(len(X1b), int)])
    # split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte
