# mlp_lots_clean_v3.py
# Previsão de receita diária (LOTS) com MLP global + forecast 1 mês à frente
# - Features de série (lags: 1,7,14,28,30,364,365; rolagens 7/28)
# - One-hot de calendário (dow, month) e do site (parking_id)
# - MLP em log1p (alvo), com early stopping e regularização
# - Ensemble com Naive-7 calibrado por site (α) + correção de viés (k) usando dezembro do ano-1
# - Regras de negócio: piso por site, "regra de zero", CAP por linha
# - Métrica: MAPE_seguro mensal + IC95 (bootstrap)
# - Forecast para o mês imediatamente após o último dia da base (simulação causal)

import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# ========= Config =========
CSV = "data/lots.csv"
OUT_DIR = "artifacts"; os.makedirs(OUT_DIR, exist_ok=True)
RND = 42

# ========= Métricas util =========
def mape_safe_daily(y_true, y_pred, floor):
    """
    MAPE diário estável (evita explosão quando y≈0).
    - Se y < piso e |ŷ| < piso -> erro 0 (dia irrelevante)
    - Caso contrário: |y-ŷ| / max(y, piso)
    """
    y  = np.asarray(y_true, float)
    yh = np.asarray(y_pred, float)
    f  = np.asarray(floor, float)
    denom = np.maximum(y, f)
    ape = np.abs(y - yh) / denom * 100.0
    ape[(y < f) & (np.abs(yh) < f)] = 0.0
    return ape

def bootstrap_ci_mean(values, n_boot=2000, alpha=0.05, seed=RND):
    """IC95% da média por bootstrap (percentis 2.5% e 97.5%)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, float)
    if len(vals) == 0:
        return np.nan, np.nan
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# ========= Painel diário + features =========
def _complete_calendar(g):
    """Preenche todas as datas (D), preserva site e zera buracos."""
    g = g.sort_values("date").set_index("date")
    full = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full)
    g["parking_id"]   = g["parking_id"].iloc[0]
    g["revenue"]      = g["revenue"].fillna(0.0)
    g["n_tx"]         = g["n_tx"].fillna(0).astype(int)
    g["ticket_medio"] = g["ticket_medio"].fillna(0.0)
    return g.reset_index().rename(columns={"index": "date"})

def _add_features(g):
    """Lags/janelas (inclui 364/365) + calendário básico."""
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
    """1) lê/limpa 2) agrega diário 3) completa calendário 4) cria features."""
    df = pd.read_csv(csv_path, sep=None, engine="python", dtype={"parking_id": str})
    for c in ["checkin_date", "checkout_date", "payment_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="ISO8601")
    # limpeza mínima
    df = df[df["paid_amount"].notna() & (df["paid_amount"] >= 0)]
    df = df[df["checkin_date"].notna() & df["checkout_date"].notna()]
    df = df[df["checkout_date"] >= df["checkin_date"]]
    # competência: payment_date (se existir), senão checkout_date
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

# ========= Forecast 1 mês à frente (simulação causal) =========
def _simulate_forward_one_month(last_date, site_hist, floors, alpha_by_site, k_by_site,
                                model, scaler, num_cols, cat_cols, train_cols,
                                cap_factors):
    """
    Gera previsões diárias para o mês imediatamente após `last_date`,
    por site (site_hist = histórico do site até last_date).
    Retorna DataFrame com colunas: date, parking_id, yhat.
    """
    site  = site_hist["parking_id"].iloc[0]
    floor = float(floors[site])
    alpha = float(alpha_by_site.get(site, 0.6))
    k     = float(k_by_site.get(site, 1.0))

    g = site_hist.sort_values("date").copy().set_index("date")
    start_fcst = (last_date + pd.Timedelta(days=1)).normalize()
    end_fcst   = (start_fcst + pd.offsets.MonthEnd(0)).normalize()
    horizon    = pd.date_range(start_fcst, end_fcst, freq="D")

    rows = []
    for d in horizon:
        # lags e rolags causais com base no "g" (que será alimentado com as previsões)
        row = {
            "parking_id": site,
            "n_tx": g["n_tx"].iloc[-1] if "n_tx" in g.columns else 0.0,
            "ticket_medio": g["ticket_medio"].iloc[-1] if "ticket_medio" in g.columns else 0.0,
            "lag_1":  g["revenue"].iloc[-1]   if len(g) >= 1   else 0.0,
            "lag_7":  g["revenue"].iloc[-7]  if len(g) >= 7   else (g["revenue"].mean() if len(g) else 0.0),
            "lag_14": g["revenue"].iloc[-14] if len(g) >= 14  else 0.0,
            "lag_28": g["revenue"].iloc[-28] if len(g) >= 28  else 0.0,
            "lag_30": g["revenue"].iloc[-30] if len(g) >= 30  else 0.0,
            "lag_364": g["revenue"].iloc[-364] if len(g) >= 364 else 0.0,
            "lag_365": g["revenue"].iloc[-365] if len(g) >= 365 else 0.0,
        }
        last7  = g["revenue"].iloc[-7:]  if len(g) >= 1 else pd.Series([0.0])
        last28 = g["revenue"].iloc[-28:] if len(g) >= 1 else pd.Series([0.0])
        row["roll_7_mean"]  = float(last7.mean())
        row["roll_28_mean"] = float(last28.mean())

        # calendário do dia d
        dow = int(pd.Timestamp(d).dayofweek)
        row["is_weekend"] = int(dow in (5,6))
        row["dow"]   = dow
        row["month"] = int(pd.Timestamp(d).month)

        # one-hot igual ao treino
        X = pd.DataFrame([row])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False).reindex(columns=train_cols, fill_value=0)
        X_num  = scaler.transform(X[num_cols])
        X_full = np.hstack([X_num, X.drop(columns=num_cols).values])

        # MLP + ensemble calibrado + correção de viés
        yhat_mlp = float(model.predict(X_full)[0])
        yhat = alpha * yhat_mlp + (1.0 - alpha) * row["lag_7"]
        yhat *= k

        # regra de zero
        if (row["lag_1"] <= 1e-6) and (row["lag_7"] <= 1e-6) and (row["roll_28_mean"] < floor):
            yhat = 0.0

        # CAP
        cap = max(
            cap_factors["roll28_mult"] * row["roll_28_mean"],
            cap_factors["lag30_mult"]  * row["lag_30"],
            cap_factors["floor_mult"]  * floor
        )
        yhat = float(np.clip(yhat, 0.0, cap))

        rows.append({"date": d, "parking_id": site, "yhat": yhat})

        # alimenta histórico com a previsão (causal) para próximos dias
        g.loc[d, "revenue"]      = yhat
        g.loc[d, "n_tx"]         = row["n_tx"]
        g.loc[d, "ticket_medio"] = row["ticket_medio"]

    return pd.DataFrame(rows)

def forecast_next_month(panel, train, model, scaler,
                        num_cols, cat_cols, alpha_by_site, k_by_site, floors,
                        cap_factors=None):
    """
    Gera previsões diárias do mês imediatamente posterior ao último dia de 'panel'.
    Salva CSVs em artifacts/ e retorna (df_diario, df_resumo).
    """
    if cap_factors is None:
        cap_factors = {"roll28_mult": 3.0, "lag30_mult": 2.0, "floor_mult": 4.0}

    last_day  = panel["date"].max()
    train_cols = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, drop_first=False).columns

    all_fcst = []
    for site, hist in panel.groupby("parking_id"):
        base_cols = ["parking_id", "date", "revenue", "n_tx", "ticket_medio"]
        site_hist = hist[base_cols].copy()
        fc = _simulate_forward_one_month(last_day, site_hist, floors, alpha_by_site, k_by_site,
                                         model, scaler, num_cols, cat_cols, train_cols,
                                         cap_factors)
        all_fcst.append(fc)

    fcst = pd.concat(all_fcst, ignore_index=True).sort_values(["parking_id","date"])
    fcst["month"] = fcst["date"].dt.to_period("M")
    monthly = (fcst.groupby(["parking_id","month"])
                    .agg(receita_prevista=("yhat","sum"),
                         dias=("yhat","size"))
                    .reset_index())

    month_str = str(fcst["month"].iloc[0])
    out_daily = os.path.join(OUT_DIR, f"forecast_{month_str}.csv")
    out_month = os.path.join(OUT_DIR, f"forecast_{month_str}_resumo.csv")
    fcst.drop(columns=["month"]).to_csv(out_daily, index=False)
    monthly.to_csv(out_month, index=False)
    print(f"✓ Forecast diário salvo em {out_daily}")
    print(f"✓ Resumo mensal salvo  em {out_month}")
    return fcst, monthly

# --- Visão mensal agregada (todos os sites) para o ano do relatório ---
def _mape_mean_ci(g):
    vals = g["MAPE_seguro_diario_%"].values
    lo, hi = bootstrap_ci_mean(vals)
    return pd.Series({
        "MAPE_seguro_%": float(vals.mean()),
        "MAPE_CI95_low": lo,
        "MAPE_CI95_high": hi
    })


# ========= Treino/Eval + previsão futura =========
def main():
    panel = build_panel(CSV)
    # PEGA 2025
    # last_year = int(panel["year"].max())
    # train = panel[panel["year"] < last_year].copy()
    # test  = panel[panel["year"] == last_year].copy()

    # OU PEGA 2024
    REPORT_YEAR = 2024
    train = panel[panel["year"] < REPORT_YEAR].copy()
    test  = panel[panel["year"] == REPORT_YEAR].copy()
    last_year = REPORT_YEAR  # o restante do código usa last_year

    assert not train.empty and not test.empty, "Faltam dados para treino/teste."

    # 1) Features
    num_cols = [
        "n_tx","ticket_medio",
        "lag_1","lag_7","lag_14","lag_28","lag_30","lag_364","lag_365",
        "roll_7_mean","roll_28_mean","is_weekend"
    ]
    cat_cols = ["dow","month","parking_id"]

    Xtr = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xte = pd.get_dummies(test [num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler   = StandardScaler()
    Xtr_num  = scaler.fit_transform(Xtr[num_cols])
    Xte_num  = scaler.transform(Xte[num_cols])
    Xtr_full = np.hstack([Xtr_num, Xtr.drop(columns=num_cols).values])
    Xte_full = np.hstack([Xte_num, Xte.drop(columns=num_cols).values])

    ytr = train["revenue"].values
    yte = test["revenue"].values

    # 2) MLP (target log1p → expm1)
    base = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        learning_rate_init=3e-4,
        alpha=1e-4,
        batch_size=128,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=40,
        random_state=RND
    )
    model = TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
    model.fit(Xtr_full, ytr)

    # ===== Calibração (dezembro do ano-1) -> α e k por site =====
    prev_dec = panel[(panel["year"] == last_year - 1) & (panel["month"] == 12)].copy()
    Xval = pd.get_dummies(prev_dec[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    Xval = Xval.reindex(columns=Xtr.columns, fill_value=0)
    Xval_num  = scaler.transform(Xval[num_cols])
    Xval_full = np.hstack([Xval_num, Xval.drop(columns=num_cols).values])

    yval_mlp  = model.predict(Xval_full)
    yval_lag7 = prev_dec["lag_7"].values

    alpha_by_site, k_by_site = {}, {}
    for site, g in prev_dec.assign(y_mlp=yval_mlp, y_l7=yval_lag7).groupby("parking_id"):
        alphas = np.linspace(0.0, 1.0, 11)
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
        k_by_site[site] = float(np.clip(k, 0.7, 1.3))

    # ===== Previsão no ano de teste =====
    yhat_mlp = model.predict(Xte_full)
    alpha_vec = test["parking_id"].map(alpha_by_site).fillna(0.6).values
    k_vec     = test["parking_id"].map(k_by_site).fillna(1.0).values
    yhat = alpha_vec * yhat_mlp + (1.0 - alpha_vec) * test["lag_7"].values
    yhat = yhat * k_vec

    # ===== Regras de negócio: piso / zero / CAP =====
    floors = {}
    for site, gtr in train.groupby("parking_id"):
        nz = gtr.loc[gtr["revenue"] > 0, "revenue"]
        floors[site] = float(max(50.0, nz.quantile(0.05) if len(nz) else 50.0))

    raw = test[num_cols + cat_cols].copy()
    raw["floor"] = raw["parking_id"].map(floors).astype(float)

    zmask = (raw["lag_1"]<=1e-6) & (raw["lag_7"]<=1e-6) & (raw["roll_28_mean"] < raw["floor"])
    yhat[zmask.values] = 0.0

    cap = np.maximum.reduce([
        3.0 * raw["roll_28_mean"].values,
        2.0 * raw["lag_30"].values,
        4.0 * raw["floor"].values
    ])
    yhat = np.clip(yhat, 0.0, cap)

    # ===== Métricas (MAPE_seguro + IC95) por mês/site =====
    out = test[["parking_id","date","revenue"]].copy()
    out["yhat"]  = yhat
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
    out_csv = os.path.join(OUT_DIR, f"mape_seguro_ic95_global_{last_year}.csv")
    monthly.to_csv(out_csv, index=False)
    print(f"\n✓ MAPE_seguro + IC95 salvo em {out_csv}\n")
    print(monthly.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


    # PARA 2024
    # MAPE mensal (média dos MAPE diários) + IC95 por mês (agregado em todos os sites)
    monthly_global = (out
        .groupby("month", as_index=False)
        .apply(_mape_mean_ci)
        .reset_index(drop=True))

    # Receita mensal prevista e real (soma diária em todos os sites)
    agg_rev = (out.groupby("month", as_index=False)
                .agg(receita_prevista=("yhat","sum"),
                    receita_real=("revenue","sum")))

    # Tabela final para o relatório
    monthly_overview = pd.merge(agg_rev, monthly_global, on="month")
    monthly_overview["month"] = monthly_overview["month"].astype(str)

    csv_overview = os.path.join(OUT_DIR, f"monthly_rev_vs_mape_{last_year}.csv")
    monthly_overview.to_csv(csv_overview, index=False)
    print(f"✓ Visão mensal agregada salva em {csv_overview}")

    # PARA 2024
    # --- Gráfico: Receita Mensal Prevista x MAPE (ano do relatório) ---
    import matplotlib.pyplot as plt

    m = monthly_overview.copy()
    m["month_num"] = m["month"].map(lambda p: int(p.split("-")[1]))
    m = m.sort_values("month_num")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bar_color = "#4C72B0"    # azul
    line_color = "#DD8452"   # laranja

    ax1.bar(m["month_num"], m["receita_prevista"], color=bar_color)
    ax1.set_xlabel("Mês (1–12)")
    ax1.set_ylabel("Receita prevista (soma)")
    ax1.tick_params(axis="y", colors=bar_color)

    ax2 = ax1.twinx()
    ax2.plot(m["month_num"], m["MAPE_seguro_%"], marker="o", color=line_color)
    ax2.set_ylabel("MAPE seguro (%)")
    ax2.tick_params(axis="y", colors=line_color)

    ax1.set_title(f"Receita Mensal Prevista x MAPE — {last_year}")
    ax1.set_xticks(range(1, 13))

    out_png = os.path.join(OUT_DIR, f"monthly_rev_vs_mape_{last_year}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"✓ Figura salva em {out_png}")



    # ===== Forecast do mês seguinte (diário + resumo) =====
    cap_cfg = {"roll28_mult": 3.0, "lag30_mult": 2.0, "floor_mult": 4.0}
    _ = forecast_next_month(panel, train, model, scaler,
                            num_cols, cat_cols, alpha_by_site, k_by_site, floors,
                            cap_factors=cap_cfg)

if __name__ == "__main__":
    main()

