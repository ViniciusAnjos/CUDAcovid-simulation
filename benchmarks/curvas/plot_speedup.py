# -*- coding: utf-8 -*-
# Graficos de SPEEDUP serial x GPU (mesmo beta, epidemia identica, MAXSIM=50).
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos", "speedup")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".png"): os.remove(os.path.join(OUT, f))

# dados (ordenados por tamanho do grid)
nomes   = ["Rocinha", "Manaus", "Brasília", "São Paulo"]
Ls      = [264, 1343, 1604, 3355]
celulas = [l*l for l in Ls]
t_ser   = [235.0, 1891.0, 2124.0, 35755.0]
t_gpu   = [128.6, 135.7, 131.3, 633.5]
speedup = [s/g for s, g in zip(t_ser, t_gpu)]
rotulos = [f"{n}\n(L={l})" for n, l in zip(nomes, Ls)]
x = np.arange(len(nomes))
COR = ["#9ecae1", "#fdae6b", "#a1d99b", "#fc9272"]

# 1. Speedup por cidade (barras)
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(x, speedup, color=COR, edgecolor="0.3", width=0.62)
ax.axhline(1.0, color="0.5", ls="--", lw=1, label="speedup = 1 (sem ganho)")
for b, sp in zip(bars, speedup):
    ax.text(b.get_x()+b.get_width()/2, sp+0.8, f"{sp:.1f}×", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(rotulos)
ax.set_ylabel("Speedup (tempo serial / tempo GPU)")
ax.set_title("Speedup da GPU sobre a CPU serial, por cidade", fontweight="bold")
ax.set_ylim(0, max(speedup)*1.15); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "speedup_por_cidade.png")); plt.close(fig)
print("salvo: speedup_por_cidade.png")

# 2. Tempos serial x GPU (barras agrupadas, escala log)
fig, ax = plt.subplots(figsize=(9, 5.5))
w = 0.38
b1 = ax.bar(x-w/2, t_ser, w, label="Serial (CPU, 1 thread)", color="#d62728", edgecolor="0.3")
b2 = ax.bar(x+w/2, t_gpu, w, label="GPU (RTX 4070 SUPER)", color="#1f77b4", edgecolor="0.3")
ax.set_yscale("log")
def rotempo(seg):
    return f"{seg/60:.0f} min" if seg >= 120 else f"{seg:.0f} s"
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.06, rotempo(b.get_height()), ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(rotulos)
ax.set_ylabel("Tempo total — MAXSIM=50 (s, escala log)")
ax.set_title("Tempo de execução: Serial × GPU por cidade", fontweight="bold")
ax.set_ylim(50, max(t_ser)*2.2); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "tempos_serial_gpu.png")); plt.close(fig)
print("salvo: tempos_serial_gpu.png")

# 3. Speedup vs tamanho do grid
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(celulas, speedup, "o-", color="#2ca02c", lw=2, ms=9)
for c, sp, n in zip(celulas, speedup, nomes):
    ax.annotate(f"{n}\n{sp:.1f}×", (c, sp), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
ax.set_xscale("log")
ax.axhline(1.0, color="0.5", ls="--", lw=1)
ax.set_xlabel("Número de células do grid (L², escala log)")
ax.set_ylabel("Speedup (serial / GPU)")
ax.set_title("O speedup cresce com o tamanho do problema", fontweight="bold")
ax.set_ylim(0, max(speedup)*1.18)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "speedup_vs_tamanho.png")); plt.close(fig)
print("salvo: speedup_vs_tamanho.png")

print("\nGraficos de speedup em:", OUT)
