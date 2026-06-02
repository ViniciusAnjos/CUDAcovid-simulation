# -*- coding: utf-8 -*-
# Graficos das curvas epidemicas das 4 cidades (GPU, R0=3.5, MAXSIM=50, 400 dias).
# Estilo padrao monografia: fonte serif, acentuacao PT, 300 dpi, layout limpo.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Estilo monografia ----------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficos")
os.makedirs(OUT, exist_ok=True)

CIDADES = {
    "ROC": ("Rocinha",   "L=264,  β=0,00490"),
    "BRA": ("Brasília", "L=1604, β=0,0995"),
    "MAN": ("Manaus",    "L=1343, β=0,0995"),
    "SP":  ("São Paulo", "L=3355, β=0,0243"),
}
COR = {"ROC": "#1f77b4", "BRA": "#2ca02c", "MAN": "#ff7f0e", "SP": "#d62728"}

def load(sigla, arq, header=True):
    p = os.path.join(BASE, sigla, arq)
    d = np.loadtxt(p, skiprows=1 if header else 0)
    return d[1:]  # pula dia 0 (artefato de reporte = zeros)

def load_prev(s):  return load(s, "epidemicsprevalence.dat")
def load_inc(s):   return load(s, "epidemicsincidence.dat")
def load_infec(s): return load(s, "Infectiousprevalence.dat", header=False)

def finalize(fig, fp):
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)
    print("salvo:", os.path.basename(fp))

# ---------- 1. Prevalencia por cidade (2 subplots) ----------
for s, (nome, params) in CIDADES.items():
    d = load_prev(s); day = d[:, 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(day, d[:, 1], label="Suscetíveis (S)", color="#1f77b4", lw=1.8)
    ax1.plot(day, d[:, 2], label="Expostos (E)", color="#9467bd", lw=1.8)
    ax1.plot(day, d[:, 5], label="Infecciosos (total)", color="#d62728", lw=1.8)
    ax1.plot(day, d[:, 8], label="Recuperados (R)", color="#2ca02c", lw=1.8)
    ax1.plot(day, d[:, 9], label="Mortes por COVID", color="#000000", lw=1.8)
    ax1.set_xlabel("Tempo (dias)"); ax1.set_ylabel("Proporção da população")
    ax1.set_title("Compartimentos principais"); ax1.set_ylim(0, 1); ax1.set_xlim(0, 400)
    ax1.legend()

    ax2.plot(day, d[:, 5], label="Infecciosos (total)", color="#d62728", lw=1.8)
    ax2.plot(day, d[:, 6], label="Hospital (H)", color="#ff7f0e", lw=1.8)
    ax2.plot(day, d[:, 7], label="UTI (ICU)", color="#8c564b", lw=1.8)
    ax2.set_xlabel("Tempo (dias)"); ax2.set_ylabel("Proporção da população")
    ax2.set_title("Carga clínica (escala ampliada)"); ax2.set_xlim(0, 400)
    ax2.legend()

    fig.suptitle(f"{nome} — prevalência ({params}, MAXSIM=50)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"{s}_prevalencia.png")); plt.close(fig)
    print("salvo:", f"{s}_prevalencia.png")

# ---------- 2. Incidencia (novas infeccoes/dia) ----------
for s, (nome, params) in CIDADES.items():
    d = load_inc(s); day = d[:, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day, d[:, 2], color=COR[s], lw=1.8, label="Novos expostos (infecções/dia)")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Novos casos por dia (proporção)")
    ax.set_title(f"{nome} — incidência diária ({params})", fontweight="bold")
    ax.set_xlim(0, 400); ax.legend()
    finalize(fig, os.path.join(OUT, f"{s}_incidencia.png"))

# ---------- 3. Infecciosos sintomaticos ----------
for s, (nome, params) in CIDADES.items():
    d = load_infec(s); day = d[:, 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day, d[:, 1], label="ISLight (leve)", color="#17becf", lw=1.8)
    ax.plot(day, d[:, 2], label="ISModerate (moderado)", color="#ff7f0e", lw=1.8)
    ax.plot(day, d[:, 3], label="ISSevere (grave)", color="#d62728", lw=1.8)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Proporção da população")
    ax.set_title(f"{nome} — infecciosos sintomáticos ({params})", fontweight="bold")
    ax.set_xlim(0, 400); ax.legend()
    finalize(fig, os.path.join(OUT, f"{s}_infecciosos_sintomaticos.png"))

# ---------- 4. MORTES ao longo do tempo, por cidade (acumuladas + diarias) ----------
for s, (nome, params) in CIDADES.items():
    dp = load_prev(s); di = load_inc(s); day = dp[:, 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(day, dp[:, 9], color="#000000", lw=1.9)
    ax1.fill_between(day, dp[:, 9], color="#000000", alpha=0.08)
    ax1.set_xlabel("Tempo (dias)"); ax1.set_ylabel("Mortes por COVID (proporção acumulada)")
    ax1.set_title("Mortalidade acumulada"); ax1.set_xlim(0, 400)
    ax2.plot(day, di[:, 9], color="#d62728", lw=1.6)
    ax2.set_xlabel("Tempo (dias)"); ax2.set_ylabel("Novas mortes por dia (proporção)")
    ax2.set_title("Mortalidade diária"); ax2.set_xlim(0, 400)
    fig.suptitle(f"{nome} — mortes por COVID ao longo do tempo ({params})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"{s}_mortes.png")); plt.close(fig)
    print("salvo:", f"{s}_mortes.png")

# ---------- 5. Comparacoes entre cidades ----------
def comparar(col, ylabel, titulo, fname, loader=load_prev, transform=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, (nome, _) in CIDADES.items():
        d = loader(s)
        y = d[:, col] if transform is None else transform(d)
        ax.plot(d[:, 0], y, label=nome, color=COR[s], lw=1.9)
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold"); ax.set_xlim(0, 400); ax.legend()
    finalize(fig, os.path.join(OUT, fname))

comparar(5, "Infecciosos (proporção)",
         "Comparação: infecciosos totais por cidade", "comparacao_infecciosos.png")
comparar(1, "Taxa de ataque acumulada (1 − S)",
         "Comparação: taxa de ataque por cidade", "comparacao_ataque.png",
         transform=lambda d: 1.0 - d[:, 1])
comparar(9, "Mortes por COVID (proporção acumulada)",
         "Comparação: mortalidade acumulada por cidade", "comparacao_mortes.png")
comparar(9, "Novas mortes por dia (proporção)",
         "Comparação: mortalidade diária por cidade", "comparacao_mortes_diarias.png",
         loader=load_inc)

print("\nTodos os graficos em:", OUT)
