# -*- coding: utf-8 -*-
# Graficos de RESULTADOS para apresentacao final. Cada figura = 2x2 (4 cidades).
# Set A (serial/):     4 figuras com a curva (serial).
# Set B (comparacao/): 4 figuras com os dois tracos -> GPU (solido) e Serial (tracejado).
# Termo "UTI" (nunca ICU). Titulos limpos.
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
OUT  = os.path.join(BASE, "graficos", "resultados")
OUT_S = os.path.join(OUT, "serial")
OUT_C = os.path.join(OUT, "comparacao")
for d in (OUT, OUT_S, OUT_C):
    os.makedirs(d, exist_ok=True)
for d in (OUT, OUT_S, OUT_C):       # limpa PNGs antigos
    for f in os.listdir(d):
        if f.endswith(".png"):
            os.remove(os.path.join(d, f))

CIDADES = [("ROC", "Rocinha (L=264)"), ("MAN", "Manaus (L=1343)"),
           ("BRA", "Brasília (L=1604)"), ("SP", "São Paulo (L=3355)")]
COR = {"ROC": "#1f77b4", "MAN": "#ff7f0e", "BRA": "#2ca02c", "SP": "#d62728"}
COMP = [(1, "S", "#1f77b4"), (5, "Infecciosos", "#d62728"),
        (8, "R", "#2ca02c"), (9, "Mortes", "#000000")]

def lser(s, arq, header=True):  # serial (mesmo beta GPU)
    return np.loadtxt(os.path.join(BASE, "validacao", s, "serial", arq), skiprows=1 if header else 0)[1:]
def lgpu(s, arq, header=True):  # gpu
    return np.loadtxt(os.path.join(BASE, s, arq), skiprows=1 if header else 0)[1:]

def painel(titulo, outdir, fname, plot_fn):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (s, nome) in zip(axes.flat, CIDADES):
        plot_fn(ax, s); ax.set_title(nome, fontweight="bold"); ax.set_xlim(0, 400)
    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, fname)); plt.close(fig)
    print("salvo:", os.path.relpath(os.path.join(outdir, fname), OUT))

# ---------------- SET A: SERIAL ----------------
def prev(ax, s):
    d = lser(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:,1], label="Suscetíveis (S)", color="#1f77b4", lw=1.7)
    ax.plot(day, d[:,2], label="Expostos (E)", color="#9467bd", lw=1.7)
    ax.plot(day, d[:,5], label="Infecciosos", color="#d62728", lw=1.7)
    ax.plot(day, d[:,8], label="Recuperados (R)", color="#2ca02c", lw=1.7)
    ax.plot(day, d[:,9], label="Mortes por COVID", color="#000000", lw=1.7)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população"); ax.set_ylim(0,1); ax.legend(fontsize=8)
def inc(ax, s):
    d = lser(s, "epidemicsincidence.dat"); day = d[:, 0]
    ax.plot(day, d[:,2], color=COR[s], lw=1.7, label="Novas infecções/dia")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Novos casos/dia (proporção)"); ax.legend()
def clin(ax, s):
    d = lser(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:,5], label="Infecciosos", color="#d62728", lw=1.7)
    ax.plot(day, d[:,6], label="Hospital (H)", color="#ff7f0e", lw=1.7)
    ax.plot(day, d[:,7], label="UTI", color="#8c564b", lw=1.7)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população"); ax.legend()
def mortes(ax, s):
    d = lser(s, "epidemicsprevalence.dat"); day = d[:, 0]
    ax.plot(day, d[:,9], color="#000000", lw=1.9); ax.fill_between(day, d[:,9], color="#000000", alpha=0.08)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Mortes por COVID (proporção acumulada)")

painel("Prevalência por cidade", OUT_S, "prevalencia.png", prev)
painel("Incidência diária por cidade", OUT_S, "incidencia.png", inc)
painel("Carga clínica por cidade (Infecciosos / Hospital / UTI)", OUT_S, "carga_clinica.png", clin)
painel("Mortalidade acumulada por cidade", OUT_S, "mortes.png", mortes)

# ---------------- SET B: COMPARACAO (GPU solido + Serial tracejado) ----------------
LT = "Sólido: Serial · Tracej.: GPU"
def c_prev(ax, s):
    g = lgpu(s,"epidemicsprevalence.dat"); se = lser(s,"epidemicsprevalence.dat"); n=min(len(g),len(se)); g,se=g[:n],se[:n]; day=g[:,0]
    for col,lab,cor in COMP:
        ax.plot(day, se[:,col], color=cor, lw=1.9, label=lab); ax.plot(day, g[:,col], color=cor, lw=1.4, ls="--")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população"); ax.set_ylim(0,1)
    ax.legend(fontsize=8, title=LT, title_fontsize=8)
def c_inc(ax, s):
    g = lgpu(s,"epidemicsincidence.dat"); se = lser(s,"epidemicsincidence.dat"); n=min(len(g),len(se)); g,se=g[:n],se[:n]; day=g[:,0]
    ax.plot(day, se[:,2], color=COR[s], lw=1.9, label="Serial"); ax.plot(day, g[:,2], color=COR[s], lw=1.4, ls="--", label="GPU")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Novos casos/dia (proporção)"); ax.legend()
def c_clin(ax, s):
    g = lgpu(s,"epidemicsprevalence.dat"); se = lser(s,"epidemicsprevalence.dat"); n=min(len(g),len(se)); g,se=g[:n],se[:n]; day=g[:,0]
    for col,lab,cor in [(5,"Infecciosos","#d62728"),(6,"Hospital (H)","#ff7f0e"),(7,"UTI","#8c564b")]:
        ax.plot(day, se[:,col], color=cor, lw=1.9, label=lab); ax.plot(day, g[:,col], color=cor, lw=1.4, ls="--")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população")
    ax.legend(fontsize=8, title=LT, title_fontsize=8)
def c_mortes(ax, s):
    g = lgpu(s,"epidemicsprevalence.dat"); se = lser(s,"epidemicsprevalence.dat"); n=min(len(g),len(se)); g,se=g[:n],se[:n]; day=g[:,0]
    ax.plot(day, se[:,9], color="#000000", lw=1.9, label="Serial"); ax.plot(day, g[:,9], color="#000000", lw=1.4, ls="--", label="GPU")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Mortes por COVID (proporção acumulada)"); ax.legend()

painel("Prevalência — Serial × GPU por cidade", OUT_C, "prevalencia.png", c_prev)
painel("Incidência diária — Serial × GPU por cidade", OUT_C, "incidencia.png", c_inc)
painel("Carga clínica — Serial × GPU por cidade (Infecciosos / Hospital / UTI)", OUT_C, "carga_clinica.png", c_clin)
painel("Mortalidade acumulada — Serial × GPU por cidade", OUT_C, "mortes.png", c_mortes)

print("\nSet A (serial) em:", OUT_S, "| Set B (comparacao) em:", OUT_C)
