# -*- coding: utf-8 -*-
# Overlay serial x GPU por cidade (curvas casadas no ataque via calibracao full-sim).
# GPU = linha solida; Serial = linha tracejada. Estilo monografia.
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
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 15, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficos")
os.makedirs(OUT, exist_ok=True)

NOMES = {"ROC": "Rocinha", "BRA": "Brasília", "MAN": "Manaus", "SP": "São Paulo"}

def load_prev(path):
    d = np.loadtxt(path, skiprows=1)
    return d[1:]  # pula dia 0

def comparar(sigla, beta_ser, beta_gpu, t_ser, t_gpu):
    nome = NOMES[sigla]
    gpu = load_prev(os.path.join(BASE, sigla, "epidemicsprevalence.dat"))
    ser = load_prev(os.path.join(BASE, "serial", sigla, "epidemicsprevalence.dat"))
    dg, ds = gpu[:, 0], ser[:, 0]
    speed = t_ser / t_gpu

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # esquerda: compartimentos principais
    ax1.plot(dg, gpu[:, 1], color="#1f77b4", lw=1.9, label="S — GPU")
    ax1.plot(ds, ser[:, 1], color="#1f77b4", lw=1.6, ls="--", label="S — Serial")
    ax1.plot(dg, gpu[:, 8], color="#2ca02c", lw=1.9, label="R — GPU")
    ax1.plot(ds, ser[:, 8], color="#2ca02c", lw=1.6, ls="--", label="R — Serial")
    ax1.plot(dg, gpu[:, 9], color="#000000", lw=1.9, label="Mortes — GPU")
    ax1.plot(ds, ser[:, 9], color="#000000", lw=1.6, ls="--", label="Mortes — Serial")
    ax1.set_xlabel("Tempo (dias)"); ax1.set_ylabel("Proporção da população")
    ax1.set_title("Compartimentos principais"); ax1.set_xlim(0, 400); ax1.set_ylim(0, 1)
    ax1.legend(ncol=3, fontsize=8.5)

    # direita: infecciosos totais (onde o timing diverge mais)
    ax2.plot(dg, gpu[:, 5], color="#d62728", lw=1.9, label="Infecciosos — GPU")
    ax2.plot(ds, ser[:, 5], color="#d62728", lw=1.6, ls="--", label="Infecciosos — Serial")
    ax2.set_xlabel("Tempo (dias)"); ax2.set_ylabel("Proporção da população")
    ax2.set_title("Infecciosos totais"); ax2.set_xlim(0, 400)
    ax2.legend()

    fig.suptitle(
        f"{nome} — Serial × GPU (ataque casado ≈ {(1-gpu[-1,1])*100:.1f}%)   |   "
        f"β_serial={beta_ser}, β_GPU={beta_gpu}   |   "
        f"tempo: Serial {t_ser:.0f}s, GPU {t_gpu:.0f}s  →  speedup {speed:.2f}×",
        fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f"{sigla}_comparacao_serial_gpu.png")
    fig.savefig(fp); plt.close(fig)
    print("salvo:", os.path.basename(fp))

if __name__ == "__main__":
    # argumentos: sigla beta_ser beta_gpu t_ser t_gpu
    s = sys.argv[1]
    comparar(s, sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
