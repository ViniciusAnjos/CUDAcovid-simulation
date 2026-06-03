# -*- coding: utf-8 -*-
# Graficos individuais por cidade para o capitulo de RESULTADOS / apresentacao final.
# Curvas limpas (prevalencia, incidencia, mortes). Sem rotulos internos.
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

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "validacao")          # dados (mesmo beta GPU)
OUT = os.path.join(BASE, "graficos", "resultados")
os.makedirs(OUT, exist_ok=True)

CIDADES = {
    "ROC": ("Rocinha",   "L=264,  β=0,0049"),
    "BRA": ("Brasília",  "L=1604, β=0,0995"),
    "MAN": ("Manaus",    "L=1343, β=0,0995"),
    "SP":  ("São Paulo", "L=3355, β=0,0243"),
}
COR = {"ROC": "#1f77b4", "BRA": "#2ca02c", "MAN": "#ff7f0e", "SP": "#d62728"}

def load(s, arq, header=True):
    d = np.loadtxt(os.path.join(SRC, s, "serial", arq), skiprows=1 if header else 0)
    return d[1:]

def lp(s): return load(s, "epidemicsprevalence.dat")
def li(s): return load(s, "epidemicsincidence.dat")
def lf(s): return load(s, "Infectiousprevalence.dat", header=False)

def fin(fig, fp):
    fig.tight_layout(); fig.savefig(fp); plt.close(fig); print("salvo:", os.path.basename(fp))

for s, (nome, params) in CIDADES.items():
    # 1. Prevalencia (2 paineis)
    d = lp(s); day = d[:, 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(day, d[:, 1], label="Suscetíveis (S)", color="#1f77b4", lw=1.9)
    a1.plot(day, d[:, 2], label="Expostos (E)", color="#9467bd", lw=1.9)
    a1.plot(day, d[:, 5], label="Infecciosos", color="#d62728", lw=1.9)
    a1.plot(day, d[:, 8], label="Recuperados (R)", color="#2ca02c", lw=1.9)
    a1.plot(day, d[:, 9], label="Mortes por COVID", color="#000000", lw=1.9)
    a1.set_xlabel("Tempo (dias)"); a1.set_ylabel("Proporção da população")
    a1.set_title("Compartimentos"); a1.set_xlim(0, 400); a1.set_ylim(0, 1); a1.legend()
    a2.plot(day, d[:, 5], label="Infecciosos", color="#d62728", lw=1.9)
    a2.plot(day, d[:, 6], label="Hospital (H)", color="#ff7f0e", lw=1.9)
    a2.plot(day, d[:, 7], label="UTI (ICU)", color="#8c564b", lw=1.9)
    a2.set_xlabel("Tempo (dias)"); a2.set_ylabel("Proporção da população")
    a2.set_title("Carga clínica"); a2.set_xlim(0, 400); a2.legend()
    fig.suptitle(f"{nome} — Prevalência ({params})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"{s}_prevalencia.png")); plt.close(fig)
    print("salvo:", f"{s}_prevalencia.png")

    # 2. Incidencia
    d = li(s); day = d[:, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day, d[:, 2], color=COR[s], lw=1.9, label="Novas infecções por dia")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Novos casos por dia (proporção)")
    ax.set_title(f"{nome} — Incidência diária ({params})", fontweight="bold")
    ax.set_xlim(0, 400); ax.legend()
    fin(fig, os.path.join(OUT, f"{s}_incidencia.png"))

    # 3. Infecciosos sintomaticos
    d = lf(s); day = d[:, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day, d[:, 1], label="Leve (ISLight)", color="#17becf", lw=1.9)
    ax.plot(day, d[:, 2], label="Moderado (ISModerate)", color="#ff7f0e", lw=1.9)
    ax.plot(day, d[:, 3], label="Grave (ISSevere)", color="#d62728", lw=1.9)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população")
    ax.set_title(f"{nome} — Infecciosos sintomáticos ({params})", fontweight="bold")
    ax.set_xlim(0, 400); ax.legend()
    fin(fig, os.path.join(OUT, f"{s}_infecciosos_sintomaticos.png"))

    # 4. Mortes (acumuladas + diarias)
    dp = lp(s); di = li(s); day = dp[:, 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(day, dp[:, 9], color="#000000", lw=2.0); a1.fill_between(day, dp[:, 9], color="#000000", alpha=0.08)
    a1.set_xlabel("Tempo (dias)"); a1.set_ylabel("Mortes por COVID (proporção acumulada)")
    a1.set_title("Mortalidade acumulada"); a1.set_xlim(0, 400)
    a2.plot(day, di[:, 9], color="#d62728", lw=1.7)
    a2.set_xlabel("Tempo (dias)"); a2.set_ylabel("Novas mortes por dia (proporção)")
    a2.set_title("Mortalidade diária"); a2.set_xlim(0, 400)
    fig.suptitle(f"{nome} — Mortes por COVID ao longo do tempo ({params})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"{s}_mortes.png")); plt.close(fig)
    print("salvo:", f"{s}_mortes.png")

print("\nGraficos de resultados em:", OUT)
