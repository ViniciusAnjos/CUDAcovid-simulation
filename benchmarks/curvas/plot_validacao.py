# -*- coding: utf-8 -*-
# Overlay serial-CORRIGIDO x GPU no MESMO Beta (apos os 2 fixes do serial).
# Serial em validacao/<cidade>/serial/, GPU em <cidade>/. Mostra que agora COINCIDEM.
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 14, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficos", "validacao")
os.makedirs(OUT, exist_ok=True)

# cidade -> (nome, beta unico p/ serial E gpu)
CIDADES = {
    "ROC": ("Rocinha",  "0,0049"),
    "BRA": ("Brasília", "0,0995"),
    "MAN": ("Manaus",   "0,0995"),
    "SP":  ("São Paulo","0,0243"),
}
COMPART = [(1, "S", "#1f77b4"), (5, "Infecciosos", "#d62728"),
           (8, "R", "#2ca02c"), (9, "Mortes", "#000000")]

def load(path):
    d = np.loadtxt(path, skiprows=1)
    return d[1:]

def disponivel(s):
    return (os.path.exists(os.path.join(BASE, s, "epidemicsprevalence.dat")) and
            os.path.exists(os.path.join(BASE, "validacao", s, "serial", "epidemicsprevalence.dat")))

feitos = []
for s, (nome, beta) in CIDADES.items():
    if not disponivel(s):
        print("pulando", s, "(faltam dados)"); continue
    gpu = load(os.path.join(BASE, s, "epidemicsprevalence.dat"))
    ser = load(os.path.join(BASE, "validacao", s, "serial", "epidemicsprevalence.dat"))
    n = min(len(gpu), len(ser)); gpu, ser = gpu[:n], ser[:n]; day = gpu[:, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for col, lab, cor in COMPART:
        ax1.plot(day, gpu[:, col], color=cor, lw=1.9, label=f"{lab} — GPU")
        ax1.plot(day, ser[:, col], color=cor, lw=1.4, ls="--", label=f"{lab} — Serial")
    ax1.set_xlabel("Tempo (dias)"); ax1.set_ylabel("Proporção da população")
    ax1.set_title("Prevalência — GPU (sólido) × Serial corrigido (tracejado)")
    ax1.set_xlim(0, 400); ax1.set_ylim(0, 1); ax1.legend(ncol=2, fontsize=8)

    for col, lab, cor in COMPART:
        ax2.plot(day, gpu[:, col] - ser[:, col], color=cor, lw=1.6, label=f"Δ {lab}")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("Tempo (dias)"); ax2.set_ylabel("Diferença (GPU − Serial)")
    ax2.set_title("Diferença (≈ 0: equivalência)"); ax2.set_xlim(0, 400)
    ax2.set_ylim(-0.1, 0.1); ax2.legend()

    atk_g, atk_s = 1-gpu[-1,1], 1-ser[-1,1]
    fig.suptitle(f"{nome} — VALIDAÇÃO Serial × GPU no MESMO β={beta}  "
                 f"(ataque GPU {atk_g*100:.1f}% / Serial {atk_s*100:.1f}%)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f"{s}_validacao_mesmobeta.png")
    fig.savefig(fp); plt.close(fig); print("salvo:", os.path.basename(fp))
    feitos.append(s)

print("\nValidacao gerada para:", feitos, "-> ", OUT)
