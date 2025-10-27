import os
import numpy as np
import pandas as pd
from main import build_panel, mape_safe_daily, bootstrap_ci_mean, RND
from main import CSV, OUT_DIR, main as train_main

# ==========================================
# 1) Re-treina modelo e obtém painel já limpo
# ==========================================
panel = build_panel(CSV)

last_year = int(panel["year"].max())   # ex: 2025
target_year = last_year - 1            # queremos 2024
print(f"\n→ Gerando forecasts para {target_year}\n")

# Rodamos o main() apenas para garantir que modelo está calibrado
train_main()

# Recarrega previsões de janeiro (já salvas pela v3)
# Agora vamos gerar forecasts mês a mês:
months = sorted(panel.loc[panel["year"] == target_year, "month"].unique())

# ==========================================
# Função para prever um mês futuro usando o mesmo pipeline
# ==========================================
from main import build_panel, mape_safe_daily, bootstrap_ci_mean

def forecast_month(year, month):
    df = panel.copy()
    df = df[df["date"].dt.to_period("M") == f"{year}-{month:02d}"]

    # Precisamos das features exatamente como na v3
    from main import num_cols, cat_cols, scaler, model, alpha_by_site, k_by_site

    X = pd.get_dummies(df[num_cols + cat_cols], columns=cat_cols, drop_first=False)
    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

    X_num = scaler.transform(X[num_cols])
    X_full = np.hstack([X_num, X.drop(columns=num_cols).values])

    yhat_mlp = model.predict(X_full)
    alpha_vec = df["parking_id"].map(alpha_by_site).fillna(0.6).values
    k_vec     = df["parking_id"].map(k_by_site).fillna(1.0).values
    yhat = alpha_vec * yhat_mlp + (1 - alpha_vec) * df["lag_7"].values
    yhat = yhat * k_vec

    # Regras piso / zero / CAP
    from main import floors
    df["floor"] = df["parking_id"].map(floors).astype(float)

    zmask = (df["lag_1"]<=1e-6) & (df["lag_7"]<=1e-6) & (df["roll_28_mean"] < df["floor"])
    yhat[zmask.values] = 0.0

    cap = np.maximum.reduce([
        3.0 * df["roll_28_mean"].values,
        2.0 * df["lag_30"].values,
        4.0 * df["floor"].values
    ])
    yhat = np.clip(yhat, 0.0, cap)

    df["yhat"] = yhat
    return df[["parking_id","date","revenue","yhat","floor"]]

# ==========================================
# 2) Gerar forecasts diários + mensais para todo 2024
# ==========================================
all_forecasts = []
for m in months:
    print(f" → Prevendo mês {m:02d}/{target_year}...")
    fm = forecast_month(target_year, m)
    fm["month"] = fm["date"].dt.to_period("M")
    all_forecasts.append(fm)

full = pd.concat(all_forecasts, ignore_index=True)
full.to_csv(f"{OUT_DIR}/forecast_{target_year}.csv", index=False)

# Resumo mensal por site
resumo = full.groupby(["parking_id","month"]).agg(
    receita_prevista=("yhat","sum"),
    receita_real=("revenue","sum")
).reset_index()
resumo.to_csv(f"{OUT_DIR}/forecast_{target_year}_resumo.csv", index=False)

# ==========================================
# 3) MAPE seguro mensal + IC95
# ==========================================
rows = []
for (site, m), g in full.groupby(["parking_id","month"]):
    ape = mape_safe_daily(g["revenue"], g["yhat"], g["floor"])
    mean_m = ape.mean()
    lo, hi = bootstrap_ci_mean(ape)
    rows.append([site, str(m), mean_m, lo, hi])

mape_2024 = pd.DataFrame(rows, columns=["parking_id","month","MAPE_seguro_%","IC_low","IC_high"])
mape_2024.to_csv(f"{OUT_DIR}/mape_seguro_ic95_global_{target_year}.csv", index=False)

print("\n✅ Forecasts e MAPE 2024 gerados!\n")
print(mape_2024)
