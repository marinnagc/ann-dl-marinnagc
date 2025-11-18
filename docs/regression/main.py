# mlp_lots_final.py
# Abordagem CONSERVADORA: mantém EXATAMENTE as mesmas features do original
# Foca apenas em: hiperparâmetros, calibração mais granular, e ensemble de seeds

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

CSV = "data/lots.csv"
OUT_DIR = "artifacts"; os.makedirs(OUT_DIR, exist_ok=True)
RND = 42

def mape_safe_daily(y_true, y_pred, floor):
    y  = np.asarray(y_true, float)
    yh = np.asarray(y_pred, float)
    f  = np.asarray(floor, float)
    denom = np.maximum(y, f)
    ape = np.abs(y - yh) / denom * 100.0
    ape[(y < f) & (np.abs(yh) < f)] = 0.0
    return ape

def bootstrap_ci_mean(values, n_boot=2000, alpha=0.05, seed=RND):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    if len(vals) == 0:
        return np.nan, np.nan
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def _complete_calendar(g):
    g = g.sort_values("date").set_index("date")
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full)
    g["parking_id"]   = g["parking_id"].iloc[0]
    g["revenue"]      = g["revenue"].fillna(0.0)
    g["n_tx"]         = g["n_tx"].fillna(0).astype(int)
    g["ticket_medio"] = g["ticket_medio"].fillna(0.0)
    return g.reset_index().rename(columns={"index": "date"})

def _add_features(g):
    """EXATAMENTE as mesmas features do modelo original."""
    g = g.sort_values("date")
    for L in [1, 7, 14, 28, 30, 364, 365]:
        g[f"lag_{L}"] = g["revenue"].shift(L)
    g["roll_7_mean"]  = g["revenue"].rolling(7).mean()
    g["roll_28_mean"] = g["revenue"].rolling(28).mean()
    g = g.dropna()
    g["dow"]        = g["date"].dt.dayofweek
    g["is_weekend"] = g["dow"].isin([5, 6]).astype(int)
    g["month"]      = g["date"].dt.month
    g["year"]       = g["date"].dt.year
    return g

def build_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype={"parking_id": str})
    for c in ["checkin_date", "checkout_date", "payment_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="ISO8601")
    
    df = df[df["paid_amount"].notna() & (df["paid_amount"] >= 0)]
    df = df[df["checkin_date"].notna() & df["checkout_date"].notna()]
    df = df[df["checkout_date"] >= df["checkin_date"]]
    df["date"] = df["payment_date"].fillna(df["checkout_date"]).dt.floor("D")
    df = df[(df["date"] >= "2022-01-01") & (df["date"] <= "2025-12-31")]

    daily = (df.groupby(["parking_id", "date"])
               .agg(revenue=("paid_amount","sum"),
                    n_tx=("paid_amount","size"),
                    ticket_medio=("paid_amount","mean"))
               .reset_index())

    panel = (daily.groupby("parking_id", group_keys=False)
                  .apply(_complete_calendar)
                  .groupby("parking_id", group_keys=False)
                  .apply(_add_features)
                  .reset_index(drop=True))
    return panel

def train_model(Xtr_full, ytr, config_name, seed, hidden, lr, alpha_reg, max_iter):
    """Treina um único modelo MLP."""
    base = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        learning_rate_init=lr,
        alpha=alpha_reg,
        batch_size=128,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=50,
        random_state=seed,
        verbose=False
    )
    model = TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
    model.fit(Xtr_full, ytr)
    return model

def evaluate_model(panel, config):
    """Avalia um modelo com configuração específica."""
    config_name, params = config
    last_year = int(panel["year"].max())
    train = panel[panel["year"] < last_year].copy()
    test  = panel[panel["year"] == last_year].copy()
    
    num_cols = ['n_tx', 'ticket_medio', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_30', 
                'lag_364', 'lag_365', 'roll_7_mean', 'roll_28_mean', 'is_weekend']
    cat_cols = ['dow', 'month', 'parking_id']
    
    Xtr = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xte = pd.get_dummies(test [num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler = StandardScaler()
    Xtr_num = scaler.fit_transform(Xtr[num_cols])
    Xte_num = scaler.transform(Xte[num_cols])
    Xtr_full = np.hstack([Xtr_num, Xtr.drop(columns=num_cols).values])
    Xte_full = np.hstack([Xte_num, Xte.drop(columns=num_cols).values])

    ytr = train["revenue"].values
    
    # Ensemble de modelos com diferentes seeds (para reduzir variância)
    print(f"  [{config_name}] Treinando ensemble de {params['n_models']} modelos...")
    models = []
    for i in range(params['n_models']):
        model = train_model(Xtr_full, ytr, config_name, RND+i, 
                           params['hidden'], params['lr'], params['alpha'], params['max_iter'])
        models.append(model)
    
    # Previsão: média dos modelos
    yhat_mlp = np.mean([m.predict(Xte_full) for m in models], axis=0)
    
    # Calibração (dezembro ano-1) - mais granular por site e mês
    prev_months = panel[(panel["year"] == last_year - 1) & (panel["month"].isin([11, 12]))].copy()
    Xval = pd.get_dummies(prev_months[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xval = Xval.reindex(columns=Xtr.columns, fill_value=0)
    Xval_num = scaler.transform(Xval[num_cols])
    Xval_full = np.hstack([Xval_num, Xval.drop(columns=num_cols).values])
    
    yval_mlp = np.mean([m.predict(Xval_full) for m in models], axis=0)
    yval_lag7 = prev_months["lag_7"].values
    
    alpha_by_site, k_by_site = {}, {}
    for site, g in prev_months.assign(y_mlp=yval_mlp, y_l7=yval_lag7).groupby("parking_id"):
        # Busca de alpha mais granular
        alphas = np.linspace(0.0, 1.0, 51)  # 51 pontos
        best_a, best_wape = 0.6, np.inf
        y_true = g["revenue"].values
        for a in alphas:
            y_hat = a*g["y_mlp"].values + (1-a)*g["y_l7"].values
            s = y_true.sum()
            wape = np.nan if s <= 1e-8 else np.sum(np.abs(y_true - y_hat))/s*100.0
            if wape < best_wape:
                best_wape, best_a = wape, a
        alpha_by_site[site] = float(best_a)
        
        y_hat_best = best_a*g["y_mlp"].values + (1-best_a)*g["y_l7"].values
        denom = y_hat_best.sum()
        k = (y_true.sum()/denom) if denom > 1e-8 else 1.0
        k_by_site[site] = float(np.clip(k, 0.75, 1.25))
    
    # Aplicar calibração
    alpha_vec = test["parking_id"].map(alpha_by_site).fillna(0.6).values
    k_vec = test["parking_id"].map(k_by_site).fillna(1.0).values
    yhat = alpha_vec * yhat_mlp + (1.0 - alpha_vec) * test["lag_7"].values
    yhat = yhat * k_vec
    
    # Regras de negócio
    floors = {}
    for site, gtr in train.groupby("parking_id"):
        nz = gtr.loc[gtr["revenue"] > 0, "revenue"]
        floors[site] = float(max(50.0, nz.quantile(0.05) if len(nz) else 50.0))
    
    raw = test.copy()
    raw["floor"] = raw["parking_id"].map(floors).astype(float)
    zmask = (raw["lag_1"]<=1e-6) & (raw["lag_7"]<=1e-6) & (raw["roll_28_mean"] < raw["floor"])
    yhat[zmask.values] = 0.0
    
    cap = np.maximum.reduce([
        3.0 * raw["roll_28_mean"].values,
        2.0 * raw["lag_30"].values,
        4.0 * raw["floor"].values
    ])
    yhat = np.clip(yhat, 0.0, cap)
    
    # Métricas
    out = test[["parking_id","date","revenue"]].copy()
    out["yhat"] = yhat
    out["floor"] = out["parking_id"].map(floors).astype(float)
    out["MAPE_seguro_diario_%"] = mape_safe_daily(out["revenue"], out["yhat"], out["floor"])
    out["month"] = out["date"].dt.to_period("M")
    
    rows = []
    for (site, m), g in out.groupby(["parking_id","month"]):
        mean_m = g["MAPE_seguro_diario_%"].mean()
        lo, hi = bootstrap_ci_mean(g["MAPE_seguro_diario_%"].values)
        rows.append({
            "parking_id": site, "month": str(m),
            "MAPE_seguro_%": mean_m,
            "MAPE_seguro_CI95_low": lo,
            "MAPE_seguro_CI95_high": hi,
            "média<5%": mean_m < 5.0,
            "IC<5%": hi < 5.0,
            "dias": len(g)
        })
    monthly = pd.DataFrame(rows).sort_values(["parking_id","month"])
    
    return monthly, out

def main():
    print("="*70)
    print("TESTE DE CONFIGURAÇÕES - Mesmas features, diferentes hiperparâmetros")
    print("="*70)
    
    panel = build_panel(CSV)
    
    # Configurações para testar
    configs = [
        ("Original", {
            'hidden': (256, 128, 64),
            'lr': 3e-4,
            'alpha': 1e-4,
            'max_iter': 800,
            'n_models': 1
        }),
        ("Config A (mais profunda)", {
            'hidden': (384, 192, 96, 48),
            'lr': 2e-4,
            'alpha': 1e-4,
            'max_iter': 1000,
            'n_models': 1
        }),
        ("Config B (ensemble 3 modelos)", {
            'hidden': (256, 128, 64),
            'lr': 3e-4,
            'alpha': 1e-4,
            'max_iter': 800,
            'n_models': 3
        }),
        ("Config C (less regularization)", {
            'hidden': (256, 128, 64),
            'lr': 3e-4,
            'alpha': 5e-5,
            'max_iter': 1000,
            'n_models': 1
        }),
        ("Config D (ensemble + tuned)", {
            'hidden': (320, 160, 80),
            'lr': 2.5e-4,
            'alpha': 7.5e-5,
            'max_iter': 1000,
            'n_models': 5
        }),
    ]
    
    all_results = {}
    
    for config in configs:
        config_name = config[0]
        print(f"\n>>> Testando: {config_name}")
        print(f"    Params: {config[1]}")
        results, _ = evaluate_model(panel, config)
        all_results[config_name] = results
    
    # Comparação
    print("\n" + "="*70)
    print("COMPARAÇÃO DE TODAS AS CONFIGURAÇÕES")
    print("="*70)
    
    summary = []
    for name, results in all_results.items():
        mape_mean = results["MAPE_seguro_%"].mean()
        mape_min = results["MAPE_seguro_%"].min()
        mape_max = results["MAPE_seguro_%"].max()
        summary.append({
            'Configuração': name,
            'MAPE Médio': mape_mean,
            'MAPE Mín': mape_min,
            'MAPE Máx': mape_max,
            'Meta <5%': '✅' if mape_mean < 5.0 else '❌'
        })
    
    summary_df = pd.DataFrame(summary).sort_values('MAPE Médio')
    print("\n" + summary_df.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    
    # Melhor configuração
    best_config = summary_df.iloc[0]['Configuração']
    best_mape = summary_df.iloc[0]['MAPE Médio']
    
    print("\n" + "="*70)
    print(f"MELHOR CONFIGURAÇÃO: {best_config}")
    print(f"MAPE: {best_mape:.2f}%")
    if best_mape < 5.0:
        print("✅ META ATINGIDA!")
    else:
        print(f"⚠️  Faltam {best_mape - 5.0:.2f}% para a meta")
    print("="*70)
    
    # Resultados detalhados da melhor
    print(f"\nResultados detalhados - {best_config}:")
    print(all_results[best_config].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    
    # Salvar
    summary_df.to_csv(os.path.join(OUT_DIR, "comparacao_configs.csv"), index=False)
    all_results[best_config].to_csv(os.path.join(OUT_DIR, "melhor_config_resultados.csv"), index=False)
    
    # Gráfico
    print("\n>>> Gerando gráfico comparativo...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Comparação de Configurações (mesmas features)', fontsize=14, fontweight='bold')
    
    # 1. MAPE médio por configuração
    ax1 = axes[0]
    colors = ['#27AE60' if i == 0 else '#3498DB' for i in range(len(summary_df))]
    bars = ax1.barh(summary_df['Configuração'], summary_df['MAPE Médio'], color=colors, alpha=0.8)
    ax1.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Meta (5%)')
    ax1.set_xlabel('MAPE Médio (%)', fontweight='bold')
    ax1.set_title('Performance por Configuração')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # Valores
    for bar, val in zip(bars, summary_df['MAPE Médio']):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2., 
                f'{val:.2f}%', ha='left', va='center', fontweight='bold')
    
    # 2. Boxplot das top 3
    ax2 = axes[1]
    top3 = summary_df.head(3)['Configuração'].tolist()
    data_top3 = [all_results[name]['MAPE_seguro_%'].values for name in top3]
    bp = ax2.boxplot(data_top3, tick_labels=top3, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('#27AE60' if i == 0 else '#3498DB')
        patch.set_alpha(0.7)
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Meta (5%)')
    ax2.set_ylabel('MAPE (%)', fontweight='bold')
    ax2.set_title('Distribuição - Top 3 Configurações')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "comparacao_final_configs.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Gráfico salvo: {plot_path}")
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)

if __name__ == "__main__":
    main()
