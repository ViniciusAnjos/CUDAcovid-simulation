# -*- coding: utf-8 -*-
# Graficos de hardware (RTX 4070 SUPER): memoria/cache vs L e impacto da otimizacao Health-SoA.
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos", "hardware")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith(".png"): os.remove(os.path.join(OUT, f))

# --- specs RTX 4070 SUPER ---
L2 = 48e6          # 48 MB
VRAM = 12e9        # 12 GB
BYTES_PERSON = 64  # GPUPerson (13 ints, align 16)
BYTES_SOA = 1      # Health compacto (unsigned char)

# ====== #1: Memoria vs L com tetos do hardware ======
L = np.logspace(2, np.log10(14000), 400)
full = (L**2) * BYTES_PERSON          # array GPUPerson (bytes)
soa  = (L**2) * BYTES_SOA             # array Health-SoA (bytes)
cidades = [("Rocinha", 264), ("Manaus", 1343), ("Brasília", 1604), ("São Paulo", 3355)]

def MB(b): return b / 1e6

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.plot(L, MB(full), color="#d62728", lw=2.2, label="Array completo (GPUPerson, 64 B/célula)")
ax.plot(L, MB(soa),  color="#1f77b4", lw=2.2, label="Array Health-SoA (1 B/célula)")
ax.axhline(MB(L2),   color="#2ca02c", ls="--", lw=1.6, label="Cache L2 = 48 MB")
ax.axhline(MB(VRAM), color="#000000", ls="--", lw=1.6, label="VRAM = 12 GB")
# L maximo (array completo cabe na VRAM)
Lmax = np.sqrt(VRAM / BYTES_PERSON)
ax.axvline(Lmax, color="0.5", ls=":", lw=1.2)
ax.text(Lmax*0.97, MB(full[0])*3, f"L máx ≈ {Lmax:.0f}\n(VRAM cheia)", ha="right", fontsize=9, color="0.3")
# cidades
for nome, l in cidades:
    ax.axvline(l, color="0.8", lw=0.8, zorder=0)
    ax.text(l, MB(VRAM)*0.45, nome, rotation=90, va="top", ha="right", fontsize=8, color="0.4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("L (lado do grid)"); ax.set_ylabel("Memória do array (MB, escala log)")
ax.set_title("Pegada de memória vs tetos da RTX 4070 SUPER", fontweight="bold")
ax.set_xlim(100, 14000); ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "memoria_vs_L.png")); plt.close(fig)
print("salvo: memoria_vs_L.png")
print(f"  L2 (48MB) comporta GPUPerson ate L={np.sqrt(L2/BYTES_PERSON):.0f}; Health-SoA ate L={np.sqrt(L2/BYTES_SOA):.0f}")

# ====== #2: Impacto da otimizacao Health-SoA (tempo/sim, Sao Paulo) ======
# Sao Paulo (memory-bound: array 721 MB >> L2, ~9.5 contatos aleatorios)
# Manaus (nao memory-bound: poucos contatos) -> sem ganho
fig, ax = plt.subplots(figsize=(8.5, 5.5))
grupos = ["São Paulo\n(L=3355, ~9,5 contatos)", "Manaus\n(L=1343, ~2 contatos)"]
antes  = [160.0, 2.84]     # s/sim sem otimizacao
depois = [11.8, 2.84]      # s/sim com Health-SoA
x = np.arange(len(grupos)); w = 0.38
b1 = ax.bar(x-w/2, antes,  w, label="Sem otimização", color="#d62728", edgecolor="0.3")
b2 = ax.bar(x+w/2, depois, w, label="Com Health-SoA", color="#1f77b4", edgecolor="0.3")
ax.set_yscale("log")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.06, f"{b.get_height():.1f}s", ha="center", fontsize=9)
# anotacao do ganho
ax.annotate(f"≈ {antes[0]/depois[0]:.0f}×", xy=(0, depois[0]), xytext=(0, antes[0]*0.5),
            ha="center", fontweight="bold", fontsize=13, color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(grupos)
ax.set_ylabel("Tempo por simulação (s, escala log)")
ax.set_title("Impacto da otimização Health-SoA (cache L2)", fontweight="bold")
ax.set_ylim(1, max(antes)*2.2); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "health_soa.png")); plt.close(fig)
print("salvo: health_soa.png")

# ====== #5: Escala com L (config fixa de Sao Paulo, MAXSIM=3) ======
Ls   = [200, 400, 800, 1600, 3200]
t_se = [2.88, 3.97, 32.19, 153.9, 607.12]   # serial s/sim
t_gp = [2.769, 2.475, 2.573, 3.375, 11.514]  # gpu s/sim
spd  = [s/g for s, g in zip(t_se, t_gp)]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
axA.plot(Ls, t_se, "o-", color="#d62728", lw=2, ms=8, label="Serial (CPU)")
axA.plot(Ls, t_gp, "s-", color="#1f77b4", lw=2, ms=8, label="GPU")
axA.axhline(min(t_gp), color="#1f77b4", ls=":", lw=1)
axA.text(210, min(t_gp)*1.15, "piso de overhead da GPU (~2,5 s/sim)", fontsize=8, color="#1f77b4")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("L (lado do grid)"); axA.set_ylabel("Tempo por simulação (s)")
axA.set_title("Tempo × L (configuração fixa)"); axA.legend()
axB.plot(Ls, spd, "o-", color="#2ca02c", lw=2, ms=8)
for l, sp in zip(Ls, spd):
    axB.annotate(f"{sp:.0f}×", (l, sp), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
axB.axhline(1.0, color="0.5", ls="--", lw=1)
axB.set_xscale("log")
axB.set_xlabel("L (lado do grid)"); axB.set_ylabel("Speedup (serial / GPU)")
axB.set_title("Speedup × L"); axB.set_ylim(0, max(spd)*1.15)
fig.suptitle("Escala com o tamanho do grid — RTX 4070 SUPER (config. São Paulo)", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "escala_L.png")); plt.close(fig)
print("salvo: escala_L.png")

# ====== #6: Block size (ocupancia), L=3200 fixo ======
bs   = [64, 128, 256, 512, 1024]
t_bs = [11.492, 11.501, 11.504, 11.534, 11.61]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(bs, t_bs, "o-", color="#1f77b4", lw=2, ms=9)
for b, t in zip(bs, t_bs):
    ax.annotate(f"{t:.2f}s", (b, t), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.set_xscale("log", base=2); ax.set_xticks(bs); ax.set_xticklabels(bs)
ax.set_xlabel("Threads por bloco"); ax.set_ylabel("Tempo por simulação (s)")
ax.set_title("Efeito do tamanho do bloco (L=3200): praticamente nulo", fontweight="bold")
ax.set_ylim(0, max(t_bs)*1.4)
ax.text(64, max(t_bs)*1.2, "Kernels limitados por memória → ocupância não é o gargalo",
        fontsize=9, color="0.3")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "blocksize.png")); plt.close(fig)
print("salvo: blocksize.png")

print("\nGraficos de hardware em:", OUT)
