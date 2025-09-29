# train_multi.py (Exercícios 3 e 4)
import numpy as np
from mlp import MLP
from data_multi import gen_multi_clusters
from sklearn.metrics import accuracy_score
import pandas as pd

Xtr, Xte, ytr, yte = gen_multi_clusters()

# Ex.3 (uma oculta)
model = MLP(layers=[(4,'relu'), (12,'relu'), (3,'linear')], task='multiclass', lr=0.05)
hist = model.fit(Xtr, ytr, epochs=400, verbose=True)
acc = (model.predict(Xte) == yte).mean()
print("Ex3 - acc:", acc)
pd.DataFrame({"y_true": yte, "y_pred": model.predict(Xte)}).to_csv("results_ex3.csv", index=False)

# Ex.4 (duas ocultas — “mais profundo”)
deep = MLP(layers=[(4,'relu'), (32,'relu'), (16,'relu'), (3,'linear')],
           task='multiclass', lr=0.05)
hist2 = deep.fit(Xtr, ytr, epochs=500, verbose=True)
acc2 = (deep.predict(Xte) == yte).mean()
print("Ex4 - acc:", acc2)
pd.DataFrame({"y_true": yte, "y_pred": deep.predict(Xte)}).to_csv("results_ex4.csv", index=False)