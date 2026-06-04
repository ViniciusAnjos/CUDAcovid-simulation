# -*- coding: utf-8 -*-
# Variabilidade entre simulacoes: bandas (media +- desvio) e histograma da taxa de ataque.
# Le os 50 prevalence_N.dat por cidade em curvas/variabilidade/<cidade>/.
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "figure.titlesize": 15, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "variabilidade")
OUT = os.path.join(BASE, "graficos", "variabilidade")
os.makedirs(OUT, exist_ok=True)
CID = [("ROC", "Rocinha (L=264)"), ("MAN", "Manaus (L=1343)"),
       ("BRA", "Brasília (L=1604)"), ("SP", "São Paulo (L=3355)")]
COR = {"ROC": "#1f77b4", "MAN": "#ff7f0e", "BRA": "#2ca02c", "SP": "#d62728"}

def load_city(s):
    fs = sorted(glob.glob(os.path.join(SRC, s, "prevalence_*.dat")))
    arr = np.stack([np.loadtxt(f) for f in fs])   # [Nsim, 401, 12]
    return arr  # col0 day, 1 S, 3 IP, 4 IA, 5 ISLight, 6 ISMod, 7 ISSev, 10 R, 11 Dead

# ---- 1. Bandas de S(t): media +- desvio ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (s, nome) in zip(axes.flat, CID):
    a = load_city(s); day = a[0, :, 0]; S = a[:, :, 1]
    m, sd = S.mean(0), S.std(0)
    ax.fill_between(day, m-sd, m+sd, color=COR[s], alpha=0.25, label="média ± desvio")
    ax.plot(day, m, color=COR[s], lw=1.9, label="média (50 sims)")
    ax.set_title(f"{nome}  (n={len(a)})", fontweight="bold")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Suscetíveis (proporção)")
    ax.set_xlim(0, 400); ax.set_ylim(0, 1); ax.legend()
fig.suptitle("Variabilidade entre simulações — banda de S(t) (média ± desvio padrão)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "bandas_S.png")); plt.close(fig)
print("salvo: bandas_S.png")

# ---- 2. Bandas dos infecciosos (IP+IA+IS*) ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (s, nome) in zip(axes.flat, CID):
    a = load_city(s); day = a[0, :, 0]
    inf = a[:, :, 3] + a[:, :, 4] + a[:, :, 5] + a[:, :, 6] + a[:, :, 7]
    m, sd = inf.mean(0), inf.std(0)
    ax.fill_between(day, np.clip(m-sd, 0, None), m+sd, color=COR[s], alpha=0.25, label="média ± desvio")
    ax.plot(day, m, color=COR[s], lw=1.9, label="média (50 sims)")
    ax.set_title(f"{nome}", fontweight="bold")
    ax.set_xlabel("Tempo (dias)"); ax.set_ylabel("Infecciosos (proporção)")
    ax.set_xlim(0, 400); ax.legend()
fig.suptitle("Variabilidade entre simulações — banda de infecciosos (IP+IA+IS)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "bandas_infecciosos.png")); plt.close(fig)
print("salvo: bandas_infecciosos.png")

# ---- 3. Histograma da taxa de ataque final (1 - S[dia 400]) ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
print("\nEstatisticas da taxa de ataque (50 sims):")
for ax, (s, nome) in zip(axes.flat, CID):
    a = load_city(s)
    atk = 1 - a[:, -1, 1]            # 1 - S no ultimo dia
    ax.hist(atk*100, bins=np.linspace(0, 100, 26), color=COR[s], edgecolor="white", alpha=0.85)
    ax.axvline(atk.mean()*100, color="black", ls="--", lw=1.5, label=f"média {atk.mean()*100:.1f}%")
    ax.set_title(f"{nome}", fontweight="bold")
    ax.set_xlabel("Taxa de ataque final (%)"); ax.set_ylabel("nº de simulações")
    ax.set_xlim(0, 100); ax.legend()
    print(f"  {nome}: media={atk.mean()*100:.1f}%  desvio={atk.std()*100:.1f}%  min={atk.min()*100:.1f}%  max={atk.max()*100:.1f}%")
fig.suptitle("Distribuição da taxa de ataque (50 simulações) — alta robustez no R0=3,5", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "histograma_ataque.png")); plt.close(fig)
print("\nsalvo: histograma_ataque.png")

# ---- 4. Contraste: incerteza perto do limiar vs robustez no R0=3.5 (Rocinha) ----
if os.path.isdir(os.path.join(SRC, "ROC_limiar")):
    al = 1 - load_city("ROC_limiar")[:, -1, 1]   # beta perto do limiar (0.0034)
    ar = 1 - load_city("ROC")[:, -1, 1]           # beta R0=3.5 (0.0049)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.hist(al*100, bins=np.linspace(0, 30, 31), color="#9467bd", edgecolor="white")
    a1.axvline(al.mean()*100, color="black", ls="--", lw=1.5, label=f"média {al.mean()*100:.1f}%")
    a1.set_title(f"Perto do limiar (β=0,0034): desvio {al.std()*100:.1f} p.p.", fontweight="bold")
    a1.set_xlabel("Taxa de ataque final (%)"); a1.set_ylabel("nº de simulações"); a1.set_xlim(0, 30); a1.legend()
    a2.hist(ar*100, bins=np.linspace(50, 65, 31), color="#1f77b4", edgecolor="white")
    a2.axvline(ar.mean()*100, color="black", ls="--", lw=1.5, label=f"média {ar.mean()*100:.1f}%")
    a2.set_title(f"Calibrado R0=3,5 (β=0,0049): desvio {ar.std()*100:.1f} p.p.", fontweight="bold")
    a2.set_xlabel("Taxa de ataque final (%)"); a2.set_ylabel("nº de simulações"); a2.set_xlim(50, 65); a2.legend()
    fig.suptitle("Rocinha — incerteza alta perto do limiar × robustez no regime calibrado", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "robustez_vs_limiar.png")); plt.close(fig)
    print(f"salvo: robustez_vs_limiar.png  (limiar desvio={al.std()*100:.1f}pp vs R0=3.5 desvio={ar.std()*100:.1f}pp)")
