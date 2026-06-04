# -*- coding: utf-8 -*-
"""Gera figura de realizacoes de Monte Carlo (curva de infecciosos) a partir
dos dados reais de variabilidade. Saida vetorial PDF para a monografia."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIDADE = "SP"
BASE = os.path.join("..", "benchmarks", "curvas", "variabilidade", CIDADE)
N_SIMS = 50
# colunas: 0=dia 1=S 2=E 3=IP 4=IA 5=ISLight 6=ISModerate 7=ISSevere 8=H 9=ICU 10=R 11=DeadCovid
COLS_INFEC = [3, 4, 5, 6, 7]  # total de infecciosos

series = []
dias_ref = None
for i in range(1, N_SIMS + 1):
    f = os.path.join(BASE, f"prevalence_{i}.dat")
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    dias = d[:, 0]
    infec = d[:, COLS_INFEC].sum(axis=1)
    series.append((dias, infec))

# trunca ao menor comprimento comum
minlen = min(len(s[1]) for s in series)
dias = series[0][0][:minlen]
mat = np.array([s[1][:minlen] for s in series])
media = mat.mean(axis=0)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
})
fig, ax = plt.subplots(figsize=(6.2, 3.8))

# realizacoes individuais
for k in range(mat.shape[0]):
    ax.plot(dias, mat[k], color="0.7", linewidth=0.5, alpha=0.6,
            zorder=1, label="Realizações individuais" if k == 0 else None)
# media
ax.plot(dias, media, color="C3", linewidth=2.2, zorder=3, label="Média (Monte Carlo)")

ax.set_xlabel("Tempo (dias)")
ax.set_ylabel("Proporção de infecciosos")
ax.set_xlim(dias.min(), dias.max())
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper right")
ax.grid(True, linewidth=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("monte_carlo.pdf")
print("monte_carlo.pdf gerado -", mat.shape[0], "realizacoes, ", minlen, "dias")
