# -*- coding: utf-8 -*-
# Graficos de benchmark: speedup e tempos serial x GPU (3 cidades). Estilo monografia.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos")

# ordenadas por tamanho do grid
nomes   = ["Rocinha\n(L=264)", "Manaus\n(L=1343)", "Brasília\n(L=1604)"]
celulas = [69696, 1803649, 2572816]
t_ser   = [142.4, 1535.0, 2124.0]
t_gpu   = [128.6, 135.7, 131.3]
speedup = [s / g for s, g in zip(t_ser, t_gpu)]
x = np.arange(len(nomes))

# 1. Speedup por cidade
fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars = ax.bar(x, speedup, color=["#9ecae1", "#fdae6b", "#a1d99b"], edgecolor="0.3", width=0.6)
ax.axhline(1.0, color="0.5", ls="--", lw=1, label="speedup = 1 (sem ganho)")
for b, sp in zip(bars, speedup):
    ax.text(b.get_x() + b.get_width()/2, sp + 0.3, f"{sp:.1f}×", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(nomes)
ax.set_ylabel("Speedup (tempo serial / tempo GPU)")
ax.set_title("Speedup da GPU × serial por cidade (MAXSIM=50, ataque casado)", fontweight="bold")
ax.set_ylim(0, max(speedup) * 1.18); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparacao_speedup.png")); plt.close(fig)
print("salvo: comparacao_speedup.png")

# 2. Tempos serial x GPU (escala log)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
w = 0.38
b1 = ax.bar(x - w/2, t_ser, w, label="Serial (CPU, /O2)", color="#d62728", edgecolor="0.3")
b2 = ax.bar(x + w/2, t_gpu, w, label="GPU (RTX 4070 SUPER)", color="#1f77b4", edgecolor="0.3")
ax.set_yscale("log")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.05, f"{b.get_height():.0f}s", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(nomes)
ax.set_ylabel("Tempo total — MAXSIM=50 (s, escala log)")
ax.set_title("Tempo de execução: serial × GPU por cidade", fontweight="bold")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparacao_tempos.png")); plt.close(fig)
print("salvo: comparacao_tempos.png")

# 3. Speedup vs tamanho do grid (escala log no eixo x)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(celulas, speedup, "o-", color="#2ca02c", lw=1.9, ms=8)
for c, sp, n in zip(celulas, speedup, ["Rocinha", "Manaus", "Brasília"]):
    ax.annotate(f"{n}\n{sp:.1f}×", (c, sp), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.set_xscale("log")
ax.axhline(1.0, color="0.5", ls="--", lw=1)
ax.set_xlabel("Número de células do grid (L², escala log)")
ax.set_ylabel("Speedup (serial / GPU)")
ax.set_title("Speedup cresce com o tamanho do problema", fontweight="bold")
ax.set_ylim(0, max(speedup) * 1.18)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparacao_speedup_vs_tamanho.png")); plt.close(fig)
print("salvo: comparacao_speedup_vs_tamanho.png")

print("\nGraficos de benchmark em:", OUT)
