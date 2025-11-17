import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# =============== utilidades de dados ===============

def set_seed(seed=42):
    np.random.seed(seed)

def make_dataset(mean0, cov0, mean1, cov1, n_per_class=1000, seed=42):
    set_seed(seed)
    X0 = np.random.multivariate_normal(mean0, cov0, n_per_class)
    X1 = np.random.multivariate_normal(mean1, cov1, n_per_class)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    # embaralha
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]

# =============== perceptron ===============

def perceptron_fit(X, y, lr=0.01, max_epochs=100, seed=0):
    """
    Treina perceptron online tradicional (sem classe).
    Retorna um dicionário com histórico para animar/avaliar.
    Perceptron online: atualiza w,b a cada amostra.
    """
    set_seed(seed) # para reprodutibilidade
    n, d = X.shape # n amostras, d dimensões (features)
    w = np.random.randn(d) * 0.01 # inicializa w pequeno
    b = 0.0                     # inicializa b = 0 (bias)

    ws, bs, accs, preds_hist = [], [], [], [] # histórico por época
    converged = False # se parou antes do max_epochs
    epochs_run = max_epochs # quantas épocas rodou de fato 

    for epoch in range(1, max_epochs + 1): 
        updates = 0
        for i in range(n):
            xi = X[i] # amostra i
            yi = y[i] # label i
            y_hat = 1 if (np.dot(xi, w) + b) >= 0 else 0 # predição (y predito binário)
            if y_hat != yi: # só atualiza se errou
                delta = (yi - y_hat)          # +1 ou -1
                w = w + lr * delta * xi # atualiza w
                b = b + lr * delta    # atualiza b
                updates += 1

        # fim da época: calcula acurácia

        scores = X @ w + b 
        '''
        @ em Python faz multiplicação de matrizes (produto matricial)
        X: geralmente é uma matriz (por exemplo, de dados de entrada, onde cada linha é 
                                    uma amostra e cada coluna é uma característica/feature).
        w: normalmente é um vetor de pesos (ou coeficientes) para cada característica.
        b: é um viés (bias), geralmente um escalar ou vetor.

        Calcula uma combinação linear dos dados de entrada (X), ponderada pelos pesos (w),
        e soma o viés (b). O resultado (scores) normalmente representa as predições de um 
        modelo linear, como regressão linear ou a camada de saída de uma rede neural simples.
        '''
        y_pred = (scores >= 0).astype(int)  # essa linha transforma os scores lineares em predições binárias (0 ou 1), 
                                            # usando a função degrau (step function) do perceptron.
        '''
        o score pode ser negativo, zero ou positivo dependendo de onde o ponto
        x está em relação à reta (no 2D) ou hiperplano (em dimensões maiores)
        O zero é exatamente a fronteira de decisão: os pontos que satisfazem
        w⋅x+b=0
        estão em cima da reta de decisão.
        Se um ponto dá score > 0, ele está de um lado da reta.
        Se dá score < 0, ele está do outro lado.
        '''
        acc = (y_pred == y).mean() # acurácia da época, média de acertos
                                   # (y_pred==y) é um array booleano, compara elemento a elemento,
                                   # True=acertou, False=errou; .mean() dá a média de acertos (True=1, False=0)
        # salva histórico
        ws.append(w.copy())
        bs.append(float(b))
        accs.append(float(acc))
        preds_hist.append(y_pred)
        # critério de parada: se não teve atualização, parou
        # como assim se nao teve atualizacao? se o perceptron acertou todas as amostras
        if updates == 0:
            converged = True # parou antes do max_epochs
            epochs_run = epoch # quantas épocas rodou de fato
            break

    return {
        "ws": ws,             # lista de w por época (np.array shape (2,))
        "bs": bs,             # lista de b por época (float)
        "accs": accs,         # acurácia por época
        "preds_hist": preds_hist,  # predições por época
        "converged": converged,
        "epochs_run": epochs_run,
        "w_final": ws[-1],
        "b_final": bs[-1],
        "y_pred_final": preds_hist[-1],
    }

# =============== plots e GIFs ===============

def compute_decision_line(w, b, x_min, x_max):
    """y = -(w1/w2)*x - b/w2 (ou linha vertical se w2 ~ 0)."""
    if abs(w[1]) > 1e-12:
        xs = np.linspace(x_min, x_max, 200)
        ys = -(w[0] / w[1]) * xs - b / w[1]
        return xs, ys, None
    elif abs(w[0]) > 1e-12:
        # vertical em x = -b/w1
        return None, None, -b / w[0]
    else:
        return None, None, None

def plot_accuracy_curve(accs, title):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(accs) + 1), accs, lw=2)
    plt.xlabel("Época"); plt.ylabel("Acurácia")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def make_gif(X, y, history, out_gif, title, fps=8):
    """
    Cria GIF animando a reta de decisão a cada época.
    Usa matplotlib.animation com writer='pillow'.
    """
    ws = history["ws"]
    bs = history["bs"]
    preds_hist = history["preds_hist"]
    epochs = len(ws)

    # limites do scatter
    pad = 1.0
    x_min, x_max = X[:,0].min() - pad, X[:,0].max() + pad
    y_min, y_max = X[:,1].min() - pad, X[:,1].max() + pad

    fig, ax = plt.subplots(figsize=(6, 6))

    def init():
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_title(title)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")

        # dados
        ax.scatter(X[y==0,0], X[y==0,1], s=8, label='Classe 0')
        ax.scatter(X[y==1,0], X[y==1,1], s=8, label='Classe 1')

        # misclassificados nessa época
        # o que é misclassificados nessa época?
        # são os pontos que o perceptron errou na predição
        y_pred = preds_hist[frame]
        mis = (y_pred != y)
        if mis.any():
            ax.scatter(X[mis,0], X[mis,1], s=28, marker='x', label='Misclass.')

        # reta de decisão
        w = ws[frame]; b = bs[frame]
        xs, ys, x_vertical = compute_decision_line(w, b, x_min, x_max)
        if xs is not None:
            ax.plot(xs, ys, lw=2, label='Limite de decisão')
        elif x_vertical is not None:
            ax.axvline(x_vertical, lw=2, label='Limite de decisão')

        # título com época e acurácia
        acc = history["accs"][frame]
        ax.set_title(f"{title} — Época {frame+1}/{epochs} · acc={acc:.3f}")
        ax.legend(loc="best")
        return []

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=epochs, interval=1000//fps, blit=False
    )

    # salva GIF
    anim.save(out_gif, writer='pillow', fps=fps)
    plt.close(fig)

# =============== um atalho para rodar experimento completo ===============

def run_experiment_with_gif(tag, mean0, cov0, mean1, cov1, n=1000, lr=0.01, max_epochs=100,
                            seed_data=42, seed_model=0, gif_name="out.gif", fps=8):
    print(f"\n===== {tag} =====")
    X, y = make_dataset(mean0, cov0, mean1, cov1, n_per_class=n, seed=seed_data)

    # distribuição (estático)
    plt.figure(figsize=(6,6))
    plt.scatter(X[y==0,0], X[y==0,1], s=8, label='Classe 0')
    plt.scatter(X[y==1,0], X[y==1,1], s=8, label='Classe 1')
    plt.title(f"{tag}: Distribuição de Dados")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend()
    plt.tight_layout(); plt.show()

    # treino
    hist = perceptron_fit(X, y, lr=lr, max_epochs=max_epochs, seed=seed_model)

    # gif da reta por época
    make_gif(X, y, hist, gif_name, f"{tag}: Reta por Época", fps=fps)

    # curva de acurácia (estático)
    plot_accuracy_curve(hist["accs"], f"{tag}: Acurácia por Época")

    # métricas no console
    mis = (hist["y_pred_final"] != y)
    print("Convergiu:", hist["converged"], "| Épocas:", hist["epochs_run"])
    print("w_final:", np.round(hist["w_final"], 6), "| b_final:", round(hist["b_final"], 6))
    print("Acurácia final:", round(hist["accs"][-1], 6))
    print("Erros:", int(mis.sum()), "de", len(y))
    print("GIF salvo em:", gif_name)

# =============== parâmetros do enunciado + execução ===============

if __name__ == "__main__":
    # Exercício 1 — separável
    EX1_mean0 = [1.5, 1.5]
    EX1_cov0  = [[0.5, 0], [0, 0.5]]
    EX1_mean1 = [5.0, 5.0]
    EX1_cov1  = [[0.5, 0], [0, 0.5]]

    # Exercício 2 — sobreposição
    EX2_mean0 = [3.0, 3.0]
    EX2_cov0  = [[1.5, 0], [0, 1.5]]
    EX2_mean1 = [4.0, 4.0]
    EX2_cov1  = [[1.5, 0], [0, 1.5]]

    LR = 0.01
    EPOCHS = 100
    N = 1000

    run_experiment_with_gif("Exercício 1", EX1_mean0, EX1_cov0, EX1_mean1, EX1_cov1,
                            n=N, lr=LR, max_epochs=EPOCHS, seed_data=123, seed_model=0,
                            gif_name="ex1.gif", fps=8)

    run_experiment_with_gif("Exercício 2", EX2_mean0, EX2_cov0, EX2_mean1, EX2_cov1,
                            n=N, lr=LR, max_epochs=EPOCHS, seed_data=456, seed_model=0,
                            gif_name="ex2.gif", fps=8)
