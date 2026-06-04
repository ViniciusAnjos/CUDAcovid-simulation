# -*- coding: utf-8 -*-
# Comparacao entre GPUs: RTX 4070 SUPER x GTX 1050 Ti (mesmo MAXSIM=5).
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
    "figure.titlesize": 15, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos", "hardware")
os.makedirs(OUT, exist_ok=True)

nomes = ["Rocinha\n(L=264)", "Manaus\n(L=1343)", "Brasília\n(L=1604)", "São Paulo\n(L=3355)"]
t_4070 = [2.53, 2.92, 2.87, 12.75]    # s/sim, MAXSIM=5
t_1050 = [21.78, 13.54, 11.83, 71.95]
ratio = [b/a for a, b in zip(t_4070, t_1050)]
x = np.arange(len(nomes))
BW_RATIO = 504.0/112.0      # banda de memoria: 4.5x
FP_RATIO = 35.0/2.1         # FP32: ~16.7x

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# painel 1: tempo/sim (log)
w = 0.38
b1 = a1.bar(x-w/2, t_4070, w, label="RTX 4070 SUPER", color="#1f77b4", edgecolor="0.3")
b2 = a1.bar(x+w/2, t_1050, w, label="GTX 1050 Ti", color="#ff7f0e", edgecolor="0.3")
a1.set_yscale("log")
for b in list(b1)+list(b2):
    a1.text(b.get_x()+b.get_width()/2, b.get_height()*1.07, f"{b.get_height():.1f}s", ha="center", fontsize=8.5)
a1.set_xticks(x); a1.set_xticklabels(nomes)
a1.set_ylabel("Tempo por simulação (s, escala log)")
a1.set_title("Tempo: 4070 SUPER × 1050 Ti (MAXSIM=5)"); a1.set_ylim(1, max(t_1050)*2.5); a1.legend()

# painel 2: razao (1050Ti / 4070S) com tetos teoricos
bars = a2.bar(x, ratio, color=["#9ecae1", "#fdae6b", "#a1d99b", "#fc9272"], edgecolor="0.3", width=0.6)
for b, r in zip(bars, ratio):
    a2.text(b.get_x()+b.get_width()/2, r+0.2, f"{r:.1f}×", ha="center", fontweight="bold")
a2.axhline(BW_RATIO, color="#d62728", ls="--", lw=1.6, label=f"razão de banda de memória ({BW_RATIO:.1f}×)")
a2.axhline(FP_RATIO, color="0.5", ls=":", lw=1.4, label=f"razão de poder de cálculo FP32 ({FP_RATIO:.0f}×)")
a2.set_xticks(x); a2.set_xticklabels(nomes)
a2.set_ylabel("Quantas vezes mais lenta (1050 Ti / 4070 SUPER)")
a2.set_title("Razão entre GPUs ≈ razão de BANDA (memory-bound)")
a2.set_ylim(0, FP_RATIO*1.1); a2.legend(loc="upper center", fontsize=9)

fig.suptitle("Comparação entre GPUs — a simulação é limitada por memória", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "comparacao_gpus.png")); plt.close(fig)
print("salvo: comparacao_gpus.png")
print(f"  razoes medidas (cidades grandes): {ratio[1]:.1f}x {ratio[2]:.1f}x {ratio[3]:.1f}x ~ banda {BW_RATIO:.1f}x (nao {FP_RATIO:.0f}x de FP32)")
