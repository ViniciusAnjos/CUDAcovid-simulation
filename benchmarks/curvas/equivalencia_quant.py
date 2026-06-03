# -*- coding: utf-8 -*-
# Quantifica a equivalencia serial x GPU (mesmo beta): RMSE e diferenca maxima por cidade.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficos", "validacao")
os.makedirs(OUT, exist_ok=True)
CID = [("ROC", "Rocinha"), ("MAN", "Manaus"), ("BRA", "Brasília"), ("SP", "São Paulo")]
# colunas: 1 S, 5 Infecciosos, 8 R, 9 Mortes
COMP = [(1, "S"), (5, "Infecc."), (8, "R"), (9, "Mortes")]

def load(p): return np.loadtxt(p, skiprows=1)[1:]

print("| Cidade | RMSE S | RMSE Infecc. | RMSE R | RMSE Mortes | máx|ΔS| | Δataque final |")
print("|--------|--------|--------------|--------|-------------|---------|---------------|")
rmseS = []
for s, nome in CID:
    g = load(os.path.join(BASE, s, "epidemicsprevalence.dat"))
    se = load(os.path.join(BASE, "validacao", s, "serial", "epidemicsprevalence.dat"))
    n = min(len(g), len(se)); g, se = g[:n], se[:n]
    r = {}
    for col, lab in COMP:
        r[lab] = np.sqrt(np.mean((g[:, col] - se[:, col])**2)) * 100  # p.p.
    maxdS = np.max(np.abs(g[:, 1] - se[:, 1])) * 100
    datk = abs((1-g[-1, 1]) - (1-se[-1, 1])) * 100
    rmseS.append(r["S"])
    print(f"| {nome} | {r['S']:.2f} | {r['Infecc.']:.3f} | {r['R']:.2f} | {r['Mortes']:.2f} | {maxdS:.1f} | {datk:.2f} |")
print("\n(valores em pontos percentuais da população; sobre os 400 dias)")

# grafico: RMSE de S(t) por cidade
fig, ax = plt.subplots(figsize=(8.5, 5))
nomes = [n for _, n in CID]
bars = ax.bar(nomes, rmseS, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="0.3", width=0.6)
for b, v in zip(bars, rmseS):
    ax.text(b.get_x()+b.get_width()/2, v+0.1, f"{v:.1f}", ha="center", fontweight="bold")
ax.axhline(2.0, color="0.5", ls="--", lw=1, label="~ruído estatístico (MAXSIM finito)")
ax.set_ylabel("RMSE de S(t) — Serial × GPU (pontos percentuais)")
ax.set_title("Diferença residual Serial × GPU no mesmo β (quanto menor, mais equivalente)", fontweight="bold", fontsize=11.5)
ax.set_ylim(0, max(rmseS)*1.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "equivalencia_rmse.png")); plt.close(fig)
print("\nsalvo: equivalencia_rmse.png")
