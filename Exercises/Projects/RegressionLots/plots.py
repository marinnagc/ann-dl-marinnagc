# plots.py
# Geração de gráficos para o projeto LOTS:
# - Forecast diário por site (mês futuro)
# - Resumo mensal de forecast (barras por site)
# - MAPE_seguro mensal com IC95 do ano de teste

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def _title_safe(s):
    return str(s).replace(":", "-").replace("/", "-").replace("\\", "-")

# ---------- PLOTS: FORECAST DIÁRIO (linha) ----------
def plot_forecast_daily(forecast_csv: str, out_dir: str = OUT_DIR):
    """
    Lê artifacts/forecast_<YYYY-MM>.csv com colunas:
      date, parking_id, yhat
    Gera:
      - plot_forecast_diario_<YYYY-MM>.png (todos os sites)
      - plot_forecast_diario_<YYYY-MM>_<site>.png (um por site)
    """
    df = pd.read_csv(forecast_csv, parse_dates=["date"])
    if not {"date", "parking_id", "yhat"}.issubset(df.columns):
        raise ValueError("CSV de forecast diário precisa ter colunas: date, parking_id, yhat.")

    month_str = str(df["date"].dt.to_period("M").iloc[0])

    # 1) Todos os sites juntos
    plt.figure(figsize=(10, 5))
    for site, g in df.groupby("parking_id"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["yhat"], marker="o", linewidth=1.5, label=str(site))
    plt.title(f"Forecast diário por site — {month_str}")
    plt.xlabel("Data")
    plt.ylabel("Receita prevista (R$)")
    plt.legend(title="parking_id")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"plot_forecast_diario_{month_str}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✓ salvo {out_path}")

    # 2) Um gráfico por site
    for site, g in df.groupby("parking_id"):
        g = g.sort_values("date")
        plt.figure(figsize=(9, 4))
        plt.plot(g["date"], g["yhat"], marker="o", linewidth=1.8)
        plt.title(f"Forecast diário — site {site} — {month_str}")
        plt.xlabel("Data")
        plt.ylabel("Receita prevista (R$)")
        plt.tight_layout()
        spath = os.path.join(out_dir, f"plot_forecast_diario_{month_str}_{_title_safe(site)}.png")
        plt.savefig(spath, dpi=160)
        plt.close()
        print(f"✓ salvo {spath}")

# ---------- PLOTS: RESUMO MENSAL (barras) ----------
def plot_forecast_monthly_bars(forecast_monthly_csv: str, out_dir: str = OUT_DIR):
    """
    Lê artifacts/forecast_<YYYY-MM>_resumo.csv com colunas:
      parking_id, month, receita_prevista, dias
    Gera:
      - plot_forecast_resumo_<YYYY-MM>.png
    """
    df = pd.read_csv(forecast_monthly_csv)
    required = {"parking_id", "month", "receita_prevista", "dias"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV de resumo mensal precisa ter colunas: {required}.")

    month_str = str(df["month"].iloc[0])
    df = df.sort_values("parking_id")

    plt.figure(figsize=(8, 4.5))
    plt.bar(df["parking_id"].astype(str), df["receita_prevista"])
    plt.title(f"Receita prevista por site — {month_str}")
    plt.xlabel("parking_id")
    plt.ylabel("Receita prevista no mês (R$)")
    for i, v in enumerate(df["receita_prevista"]):
        plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"plot_forecast_resumo_{month_str}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"✓ salvo {out_path}")

# ---------- PLOTS: MAPE seguro com IC95 (barras + erro) ----------
def plot_mape_ic95(mape_csv: str, out_dir: str = OUT_DIR):
    """
    Lê artifacts/mape_seguro_ic95_global_<ANO>.csv com colunas:
      parking_id, month, MAPE_seguro_%, MAPE_seguro_CI95_low, MAPE_seguro_CI95_high, ...
    Gera:
      - plot_mape_ic95_<ANO>.png (barra com barras de erro por site para cada mês do arquivo)
    """
    df = pd.read_csv(mape_csv)
    required = {"parking_id","month","MAPE_seguro_%","MAPE_seguro_CI95_low","MAPE_seguro_CI95_high"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV de MAPE IC95 precisa ter colunas: {required}.")

    # inferir ano do primeiro mês
    first_month = str(df["month"].iloc[0])
    year = first_month.split("-")[0] if "-" in first_month else "xxxx"

    # vamos plotar um gráfico por mês presente no CSV
    for month, g in df.groupby("month"):
        g = g.sort_values("parking_id")
        m = g["MAPE_seguro_%"].values
        err_low  = m - g["MAPE_seguro_CI95_low"].values
        err_high = g["MAPE_seguro_CI95_high"].values - m
        x = range(len(g))

        plt.figure(figsize=(8, 4.5))
        plt.bar([str(s) for s in g["parking_id"]], m, yerr=[err_low, err_high],
                capsize=4, ecolor="black")
        plt.title(f"MAPE seguro (IC95) — {month}")
        plt.xlabel("parking_id")
        plt.ylabel("MAPE (%)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"plot_mape_ic95_{year}_{str(month)}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"✓ salvo {out_path}")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Gera gráficos do projeto LOTS.")
    parser.add_argument("--forecast", help="Caminho do forecast_<YYYY-MM>.csv")
    parser.add_argument("--forecast-monthly", help="Caminho do forecast_<YYYY-MM>_resumo.csv")
    parser.add_argument("--mape", help="Caminho do mape_seguro_ic95_global_<ANO>.csv")
    parser.add_argument("--out", default=OUT_DIR, help="Diretório de saída (default: artifacts)")
    args = parser.parse_args()

    if args.forecast:
        plot_forecast_daily(args.forecast, args.out)
    if args.forecast_monthly:
        plot_forecast_monthly_bars(args.forecast_monthly, args.out)
    if args.mape:
        plot_mape_ic95(args.mape, args.out)

    if not (args.forecast or args.forecast_monthly or args.mape):
        print("Nada para fazer. Informe pelo menos um dos argumentos: "
              "--forecast, --forecast-monthly, --mape.")

if __name__ == "__main__":
    main()
