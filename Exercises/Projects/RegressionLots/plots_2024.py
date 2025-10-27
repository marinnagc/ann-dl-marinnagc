import pandas as pd
import matplotlib.pyplot as plt

resumo = pd.read_csv("artifacts/forecast_2024_resumo.csv")
mape = pd.read_csv("artifacts/mape_seguro_ic95_global_2024.csv")

# ----- Receita mensal prevista por site (2024) -----
pivot = resumo.pivot(index="month", columns="parking_id", values="receita_prevista")
pivot.plot(kind="line", marker="o", figsize=(14,6))
plt.title("Receita Mensal Prevista — 2024")
plt.ylabel("R$ (soma no mês)")
plt.savefig("artifacts/plot_receita_mensal_2024.png")

# ----- MAPE mensal por site (2024) -----
pivot_mape = mape.pivot(index="month", columns="parking_id", values="MAPE_seguro_%")
pivot_mape.plot(kind="bar", figsize=(14,6))
plt.title("MAPE seguro mensal — 2024")
plt.ylabel("MAPE (%)")
plt.savefig("artifacts/plot_mape_mensal_2024.png")

print("✅ Gráficos salvos em artifacts/")
