# plot_all.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)

DATA_DIR = Path("data")  # ajuste se precisar

# ===================== utils =====================
def read_loss_csv(path: Path):
    if not path.exists():
        print(f"[SKIP] {path} não encontrado.")
        return None
    df = pd.read_csv(path, engine="python")
    for col in ["loss", "Loss", "value", "Value"]:
        if col in df.columns:
            return df[col].astype(float).values
    raise ValueError(f"{path}: coluna de loss não encontrada (esperado 'loss').")

def read_results_csv(path: Path, expected_labels=None):
    if not path.exists():
        print(f"[SKIP] {path} não encontrado.")
        return None, None
    df = pd.read_csv(path, engine="python", skip_blank_lines=True)
    cols = {c.strip().lower(): c for c in df.columns}
    if "y_true" not in cols or "y_pred" not in cols:
        raise ValueError(f"{path}: precisa ter colunas 'y_true' e 'y_pred'.")
    yt = df[cols["y_true"]].astype(str).str.strip()
    yp = df[cols["y_pred"]].astype(str).str.strip()
    yt = pd.to_numeric(yt, errors="coerce")
    yp = pd.to_numeric(yp, errors="coerce")
    mask = yt.notna() & yp.notna()
    dropped = len(yt) - mask.sum()
    if dropped > 0:
        print(f"[WARN] {path}: {dropped} linha(s) descartada(s) por NaN/valores inválidos.")
    yt = yt[mask].astype(int).values
    yp = yp[mask].astype(int).values
    if expected_labels is not None:
        u_true = sorted(set(np.unique(yt)))
        u_pred = sorted(set(np.unique(yp)))
        if set(u_true) - set(expected_labels) or set(u_pred) - set(expected_labels):
            print(f"[WARN] {path}: valores fora do esperado. true={u_true} pred={u_pred} esperado={expected_labels}")
    return yt, yp

def read_proba_csv(path: Path):
    # opcional: para ROC binário (colunas: y_true, y_score)
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if not {"y_true","y_score"}.issubset(df.columns):
        return None, None
    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_score = pd.to_numeric(df["y_score"], errors="coerce")
    mask = y_true.notna() & y_score.notna()
    return y_true[mask].astype(int).values, y_score[mask].values

def save_and_show(fname):
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"[OK] salvo: {fname}")
    plt.show()

# ===================== plots =====================
def plot_loss_comparativo():
    files = [DATA_DIR/"hist_ex2.csv", DATA_DIR/"hist_ex3.csv", DATA_DIR/"hist_ex4.csv"]
    labels = ["Ex2 (binário)", "Ex3 (3 classes)", "Ex4 (profundo)"]
    plt.figure(figsize=(8,4.5))
    any_plotted = False
    for f, lbl in zip(files, labels):
        hist = read_loss_csv(f)
        if hist is None: 
            continue
        plt.plot(range(1, len(hist)+1), hist, label=lbl)
        any_plotted = True
    if not any_plotted:
        print("[SKIP] Nenhum CSV de loss encontrado para o comparativo.")
        return
    plt.title("Training Loss - Exercícios 2 a 4")
    plt.xlabel("Epoch checkpoint")
    plt.ylabel("Loss")
    plt.grid(True); plt.legend()
    save_and_show("loss_curves_all.png")

def plot_loss_individual(name, csv_path):
    hist = read_loss_csv(csv_path)
    if hist is None: 
        return
    plt.figure(figsize=(7,4))
    plt.plot(range(1, len(hist)+1), hist)
    plt.title(f"Training Loss - {name}")
    plt.xlabel("Epoch checkpoint"); plt.ylabel("Loss")
    plt.grid(True)
    save_and_show(f"loss_{name.lower().replace(' ','_')}.png")

def plot_cm(name, csv_path, labels):
    y_true, y_pred = read_results_csv(csv_path, expected_labels=labels)
    if y_true is None:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title(f"{name} - Confusion Matrix")
    save_and_show(f"cm_{name.lower().replace(' ','_')}.png")
    # resumo txt
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
    (Path(".")/f"metrics_{name.lower().replace(' ','_')}.txt").write_text(rep, encoding="utf-8")
    print(f"[OK] salvo: metrics_{name.lower().replace(' ','_')}.txt")

def plot_roc_ex2():
    # precisa de proba_ex2.csv com colunas: y_true,y_score
    y_true, y_score = read_proba_csv(DATA_DIR/"proba_ex2.csv")
    if y_true is None:
        print("[SKIP] ROC Ex2: arquivo 'proba_ex2.csv' não encontrado (opcional).")
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Ex2 - ROC Curve (binário)")
    plt.legend(); plt.grid(True)
    save_and_show("roc_ex2.png")

# ===================== main =====================
if __name__ == "__main__":
    # Comparativo de loss
    plot_loss_comparativo()

    # Loss individual
    plot_loss_individual("Ex2", DATA_DIR/"hist_ex2.csv")
    plot_loss_individual("Ex3", DATA_DIR/"hist_ex3.csv")
    plot_loss_individual("Ex4", DATA_DIR/"hist_ex4.csv")

    # Matrizes de confusão (gera as que existirem)
    plot_cm("Ex2", DATA_DIR/"results_ex2.csv", labels=[0,1])
    plot_cm("Ex3", DATA_DIR/"results_ex3.csv", labels=[0,1,2])
    plot_cm("Ex4", DATA_DIR/"results_ex4.csv", labels=[0,1,2])

    # ROC binário (opcional)
    plot_roc_ex2()
