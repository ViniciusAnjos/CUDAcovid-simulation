# -*- coding: utf-8 -*-
# Graficos de RESULTADOS para apresentacao final.
# 4 figuras (uma por tipo), cada uma 2x2 com as 4 cidades. Termo "UTI" (nunca ICU).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.titlesize": 15, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "validacao")
OUT = os.path.join(BASE, "graficos", "resultados")
os.makedirs(OUT, exist_ok=True)
# remove os PNGs individuais antigos
for f in os.listdir(OUT):
    if f.endswith(".png"):
        os.remove(os.path.join(OUT, f))

# ordem dos paineis (por tamanho do grid)
CIDADES = [("ROC", "Rocinha (L=264)"),
           ("MAN", "Manaus (L=1343)"),
           ("BRA", "Brasília (L=1604)"),
           ("SP",  "São Paulo (L=3355)")]
COR = {"ROC": "#1f77b4", "MAN": "#ff7f0e", "BRA": "#2ca02c", "SP": "#d62728"}

def load(s, arq, header=True):
    d = np.loadtxt(os.path.join(SRC, s, "serial", arq), skiprows=1 if header else 0)
    return d[1:]

def painel(titulo_fig, fname, plot_fn):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (s, nome) in zip(axes.flat, CIDADES):
        plot_fn(ax, s)
        ax.set_title(nome, fontweight="bold"); ax.set_xlim(0, 400)
    fig.suptitle(titulo_fig, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)
    print("salvo:", fname)

# 1. Prevalencia (compartimentos)
def f_prev(ax, s):
    d = load(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:, 1], label="Suscetíveis (S)", color="#1f77b4", lw=1.7)
    ax.plot(day, d[:, 2], label="Expostos (E)", color="#9467bd", lw=1.7)
    ax.plot(day, d[:, 5], label="Infecciosos", color="#d62728", lw=1.7)
    ax.plot(day, d[:, 8], label="Recuperados (R)", color="#2ca02c", lw=1.7)
    ax.plot(day, d[:, 9], label="Mortes por COVID", color="#000000", lw=1.7)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
painel("Prevalência por cidade", "prevalencia.png", f_prev)

# 2. Incidencia (novas infeccoes/dia)
def f_inc(ax, s):
    d = load(s, "epidemicsincidence.dat"); day = d[:, 0]
    ax.plot(day, d[:, 2], color=COR[s], lw=1.7, label="Novas infecções/dia")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Novos casos/dia (proporção)")
    ax.legend()
painel("Incidência diária por cidade", "incidencia.png", f_inc)

# 3. Carga clinica (Infecciosos / Hospital / UTI)
def f_clin(ax, s):
    d = load(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:, 5], label="Infecciosos", color="#d62728", lw=1.7)
    ax.plot(day, d[:, 6], label="Hospital (H)", color="#ff7f0e", lw=1.7)
    ax.plot(day, d[:, 7], label="UTI", color="#8c564b", lw=1.7)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população")
    ax.legend()
painel("Carga clínica por cidade (Infecciosos / Hospital / UTI)", "carga_clinica.png", f_clin)

# 4. Mortes ao longo do tempo (acumulada)
def f_mortes(ax, s):
    d = load(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:, 9], color="#000000", lw=1.9)
    ax.fill_between(day, d[:, 9], color="#000000", alpha=0.08)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Mortes por COVID (proporção acumulada)")
painel("Mortalidade acumulada por cidade", "mortes.png", f_mortes)

print("\n4 figuras (2x2 cidades) em:", OUT)
