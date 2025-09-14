# Perceptron — organizado pelos 7 passos do "Perceptron Training Process"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ============================================================
# PASSO 1 — Initialize Weights and Bias
# ============================================================

def step1_init_params(n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=n_features)  # pequenos valores
    b = 0.0
    return w, b

# ============================================================
# PASSO 2 — Provide Training Data
# (gera dataset 2D com multivariada; retorna X (2D), y (0/1))
# ============================================================

def step2_make_dataset(mean0, cov0, mean1, cov1, n_per_class=1000, seed=42):
    rng = np.random.default_rng(seed)
    X0 = rng.multivariate_normal(mean0, cov0, n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, n_per_class)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])

    # embaralha
    idx = rng.permutation(len(X)) # embaralha pq as classes estão ordenadas
    return X[idx], y[idx]

# ============================================================
# PASSO 3 — Forward Pass: Compute Prediction
# (score linear + função degrau)
# ============================================================

def step3_forward_predict(X, w, b):
    scores = X @ w + b                # w·x + b
    y_pred = (scores >= 0).astype(int)  # step(scores)
    return scores, y_pred

# ============================================================
# PASSO 4 — Compute Error
# (y - y_hat) por amostra
# ============================================================

def step4_error(y_true, y_pred):
    return y_true - y_pred  # ∈ {-1, 0, +1}

# ============================================================
# PASSO 5 — Update Weights and Bias (Perceptron Learning Rule)
# ============================================================

def step5_update(w, b, x, err, lr):
    # lr: learning rate (alpha)
    # Apenas atualiza quando err != 0
    if err != 0:
        w = w + lr * err * x
        b = b + lr * err
        updated = True
    else:
        updated = False
    return w, b, updated

# ============================================================
# PASSO 6 — Iterate (épocas) + PASSO 7 — Convergence / Stop
# (treino completo; retorna histórico por época)
# ============================================================

def train_perceptron(X, y, lr=0.01, max_epochs=100, seed_params=0):
    n, d = X.shape
    w, b = step1_init_params(d, seed=seed_params)

    accs, ws, bs, preds_hist = [], [], [], []
    converged = False
    epochs_run = max_epochs

    idx = np.arange(n)  # mesma ordem em todas as épocas

    for epoch in range(1, max_epochs + 1):
        updates = 0

        for i in idx:
            xi, yi = X[i], y[i]
            _, yhat = step3_forward_predict(xi[None, :], w, b)
            err = step4_error(yi, yhat[0])

            w, b, did_update = step5_update(w, b, xi, err, lr)
            if did_update:
                updates += 1

        # métricas por época
        _, y_pred_full = step3_forward_predict(X, w, b)
        acc = (y_pred_full == y).mean()
        accs.append(float(acc))
        ws.append(w.copy())
        bs.append(float(b))
        preds_hist.append(y_pred_full)

        if updates == 0:  # convergiu
            converged = True
            epochs_run = epoch
            break

    return {
        "ws": ws,
        "bs": bs,
        "accs": accs,
        "preds_hist": preds_hist,
        "converged": converged,
        "epochs_run": epochs_run,
        "w_final": ws[-1],
        "b_final": bs[-1],
        "y_pred_final": preds_hist[-1],
    }


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plot_scatter(X, y, title):
    plt.figure(figsize=(6,6))
    plt.scatter(X[y==0,0], X[y==0,1], s=8, label="Classe 0")
    plt.scatter(X[y==1,0], X[y==1,1], s=8, label="Classe 1")
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
    plt.tight_layout(); plt.show()

def decision_line_points(w, b, x_min, x_max):
    if abs(w[1]) > 1e-12:
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0]/w[1]) * xs - b / w[1]
        return xs, ys, None
    elif abs(w[0]) > 1e-12:
        return None, None, -b / w[0]  # linha vertical x = -b/w1
    else:
        return None, None, None

def plot_accuracy(accs, title):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1, len(accs)+1), accs, lw=2)
    plt.xlabel("Época"); plt.ylabel("Acurácia"); plt.title(title)
    plt.tight_layout(); plt.show()

def gif_decision_by_epoch(X, y, hist, out_gif, title, fps=8):
    ws, bs, accs, preds_hist = hist["ws"], hist["bs"], hist["accs"], hist["preds_hist"]
    T = len(ws)

    pad = 1.0
    x_min, x_max = X[:,0].min()-pad, X[:,0].max()+pad
    y_min, y_max = X[:,1].min()-pad, X[:,1].max()+pad

    fig, ax = plt.subplots(figsize=(6,6))

    def init():
        ax.clear()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.scatter(X[y==0,0], X[y==0,1], s=8, label="Classe 0")
        ax.scatter(X[y==1,0], X[y==1,1], s=8, label="Classe 1")

        # misclassificados nessa época
        y_pred = preds_hist[frame]
        mis = (y_pred != y)
        if mis.any():
            ax.scatter(X[mis,0], X[mis,1], s=28, marker="x", label="Misclass.")

        # reta
        w, b = ws[frame], bs[frame]
        xs, ys, xv = decision_line_points(w, b, x_min, x_max)
        if xs is not None:
            ax.plot(xs, ys, lw=2, label="Limite de decisão")
        elif xv is not None:
            ax.axvline(xv, lw=2, label="Limite de decisão")

        ax.set_title(f"{title} — Época {frame+1}/{T} · acc={accs[frame]:.3f}")
        ax.legend(loc="best")
        return []

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=T, interval=1000//fps, blit=False)
    anim.save(out_gif, writer="pillow", fps=fps)
    plt.close(fig)

# ============================================================
# EXECUÇÃO: Ex.1 (separável) e Ex.2 (sobreposição)
# ============================================================

def run_experiment(tag, mean0, cov0, mean1, cov1, n=1000, eta=0.01, max_epochs=100,
                   seed_data=42, seed_params=0, gif_name="out.gif", fps=8):
    print(f"\n=== {tag} ===")
    X, y = step2_make_dataset(mean0, cov0, mean1, cov1, n_per_class=n, seed=seed_data)

    # Visualização da distribuição
    plot_scatter(X, y, f"{tag}: Distribuição de Dados")

    # Treinamento (Passos 3–7 ocorrendo dentro de train_perceptron)
    hist = train_perceptron(X, y, eta=eta, max_epochs=max_epochs, seed_params=seed_params, reshuffle_each_epoch=True)

    # GIF da reta por época
    gif_decision_by_epoch(X, y, hist, gif_name, f"{tag}: Reta por Época", fps=fps)

    # Curva de acurácia
    plot_accuracy(hist["accs"], f"{tag}: Acurácia por Época")

    # Métricas no console
    print("Convergiu:", hist["converged"], "| Épocas:", hist["epochs_run"])
    print("w_final:", np.round(hist["w_final"], 6), "| b_final:", round(hist["b_final"], 6))
    print("Acurácia final:", round(hist["accs"][-1], 6))
    print("GIF salvo em:", gif_name)

if __name__ == "__main__":
    # Exercício 1 — separável (médias distantes, var = 0.5)
    EX1_mean0 = [1.5, 1.5]
    EX1_cov0  = [[0.5, 0], [0, 0.5]]
    EX1_mean1 = [5.0, 5.0]
    EX1_cov1  = [[0.5, 0], [0, 0.5]]

    # Exercício 2 — sobreposição (médias próximas, var = 1.5)
    EX2_mean0 = [3.0, 3.0]
    EX2_cov0  = [[1.5, 0], [0, 1.5]]
    EX2_mean1 = [4.0, 4.0]
    EX2_cov1  = [[1.5, 0], [0, 1.5]]

    ETA = 0.01
    EPOCHS = 100
    N = 1000

    run_experiment("Exercício 1", EX1_mean0, EX1_cov0, EX1_mean1, EX1_cov1,
                   n=N, eta=ETA, max_epochs=EPOCHS, seed_data=123, seed_params=0,
                   gif_name="ex1.gif", fps=8)

    run_experiment("Exercício 2", EX2_mean0, EX2_cov0, EX2_mean1, EX2_cov1,
                   n=N, eta=ETA, max_epochs=EPOCHS, seed_data=456, seed_params=0,
                   gif_name="ex2.gif", fps=8)
