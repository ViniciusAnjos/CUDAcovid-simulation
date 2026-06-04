# -*- coding: utf-8 -*-
# Prevalencia SERIAL x GPU na mesma imagem + painel de diferenca (GPU - serial) + metricas.
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
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.titlesize": 14, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficos")
os.makedirs(OUT, exist_ok=True)

CIDADES = {
    "ROC": ("Rocinha",  0.0137, 0.0049),
    "BRA": ("Brasília", 0.142,  0.0995),
    "MAN": ("Manaus",   0.142,  0.0995),
}
# colunas: 0 day 1 S 2 E 3 IP 4 IA 5 Infec 6 H 7 ICU 8 R 9 Dead
COMPART = [(1, "S", "#1f77b4"), (5, "Infecciosos", "#d62728"),
           (8, "R", "#2ca02c"), (9, "Mortes", "#000000")]

def load(path):
    d = np.loadtxt(path, skiprows=1)
    return d[1:]

linhas_doc = []
for s, (nome, bser, bgpu) in CIDADES.items():
    gpu = load(os.path.join(BASE, s, "epidemicsprevalence.dat"))
    ser = load(os.path.join(BASE, "serial", s, "epidemicsprevalence.dat"))
    n = min(len(gpu), len(ser)); gpu, ser = gpu[:n], ser[:n]
    day = gpu[:, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # painel 1: overlay
    for col, lab, cor in COMPART:
        ax1.plot(day, gpu[:, col], color=cor, lw=1.9, label=f"{lab} — GPU")
        ax1.plot(day, ser[:, col], color=cor, lw=1.5, ls="--", label=f"{lab} — Serial")
    ax1.set_xlabel("Tempo (dias)"); ax1.set_ylabel("Proporção da população")
    ax1.set_title("Prevalência — GPU (sólido) × Serial (tracejado)")
    ax1.set_xlim(0, 400); ax1.set_ylim(0, 1); ax1.legend(ncol=2, fontsize=8)

    # painel 2: diferenca GPU - serial
    for col, lab, cor in COMPART:
        ax2.plot(day, gpu[:, col] - ser[:, col], color=cor, lw=1.7, label=f"Δ {lab}")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("Tempo (dias)"); ax2.set_ylabel("Diferença (GPU − Serial)")
    ax2.set_title("Diferença entre os resultados"); ax2.set_xlim(0, 400); ax2.legend()

    fig.suptitle(f"{nome} — Prevalência Serial × GPU (β_ser={bser}, β_GPU={bgpu})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fp = os.path.join(OUT, f"{s}_prevalencia_serial_vs_gpu.png")
    fig.savefig(fp); plt.close(fig); print("salvo:", os.path.basename(fp))

    # ---- metricas de diferenca ----
    dS = gpu[:, 1] - ser[:, 1]
    rmseS = np.sqrt(np.mean(dS**2))
    imax = np.argmax(np.abs(dS))
    maxdS, maxday = dS[imax], int(day[imax])
    # picos de infecciosos
    gi, si = np.argmax(gpu[:, 5]), np.argmax(ser[:, 5])
    pg, pgd = gpu[gi, 5], int(day[gi]); ps, psd = ser[si, 5], int(day[si])
    # finais (dia 400)
    atk_g, atk_s = 1 - gpu[-1, 1], 1 - ser[-1, 1]
    r_g, r_s = gpu[-1, 8], ser[-1, 8]
    d_g, d_s = gpu[-1, 9], ser[-1, 9]
    linhas_doc.append(
        f"| {nome} | {atk_g*100:.1f}% / {atk_s*100:.1f}% (Δ{(atk_g-atk_s)*100:+.1f}) "
        f"| {d_g*100:.2f}% / {d_s*100:.2f}% (Δ{(d_g-d_s)*100:+.2f}) "
        f"| {pg*100:.2f}%@d{pgd} / {ps*100:.2f}%@d{psd} (Δdia {pgd-psd:+d}) "
        f"| {rmseS*100:.1f} p.p. | {maxdS*100:+.1f} p.p. @d{maxday} |")
    print(f"\n[{nome}] ataque GPU/ser={atk_g:.4f}/{atk_s:.4f}  mortes={d_g:.4f}/{d_s:.4f}")
    print(f"   pico infec GPU={pg:.4f}@{pgd}  ser={ps:.4f}@{psd}")
    print(f"   RMSE(S)={rmseS:.4f}  max|ΔS|={maxdS:+.4f}@dia{maxday}")

print("\n--- LINHAS P/ DOC (markdown) ---")
print("| Cidade | Ataque GPU/Ser (Δp.p.) | Mortes GPU/Ser (Δp.p.) | Pico infec GPU/Ser (Δdia) | RMSE S(t) | Máx |ΔS| |")
print("|--------|------------------------|------------------------|---------------------------|-----------|---------|")
for l in linhas_doc:
    print(l)
