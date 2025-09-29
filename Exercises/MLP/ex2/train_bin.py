# train_bin.py (Exercício 2)
import numpy as np
from mlp import MLP
from data_binario import gen_binary_clusters
from sklearn.metrics import accuracy_score, confusion_matrix

Xtr, Xte, ytr, yte = gen_binary_clusters()
model = MLP(layers=[(2,'tanh'), (8,'tanh'), (1,'sigmoid')], task='binary', lr=0.1)
hist = model.fit(Xtr, ytr, epochs=300, verbose=True)

yhat = model.predict(Xte)
acc = accuracy_score(yte, yhat)
cm  = confusion_matrix(yte, yhat)
print("Test accuracy:", acc)
print("Confusion matrix:\n", cm)
