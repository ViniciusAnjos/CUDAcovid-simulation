# -*- coding: utf-8 -*-
# Profile: onde o tempo e gasto (serial por funcao, GPU por kernel).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
    "font.size": 11, "axes.titlesize": 13, "figure.titlesize": 15,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 120,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graficos", "hardware")
os.makedirs(OUT, exist_ok=True)

# --- dados medidos (s) ---
# Serial (config SP, L=800, MAXSIM=2)
ser = {"Sfunc (contatos)": 60.994, "Updatefunc": 3.714,
       "Outros (E/IP/IS/H/ICU/Rec)": 0.285+0.655+0.131+0.009+0.001+3.878}
# GPU (config SP, L=3355, MAXSIM=3)
gpu = {"S_kernel (contatos)": 21.914, "update_kernel": 6.960, "IP_kernel (spreadInfection)": 3.051,
       "Outros (E/IS/H/ICU/bordas)": 1.193+1.135+0.985+0.967+0.053+0.032}

cores_ser = ["#d62728", "#1f77b4", "#bdbdbd"]
cores_gpu = ["#d62728", "#1f77b4", "#9467bd", "#bdbdbd"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
def pie(ax, d, cores, titulo):
    vals = list(d.values()); labs = list(d.keys()); tot = sum(vals)
    w, _, _ = ax.pie(vals, colors=cores, startangle=90, counterclock=False,
                     autopct=lambda p: f"{p:.0f}%", pctdistance=0.72,
                     wedgeprops=dict(edgecolor="white", linewidth=1.5),
                     textprops=dict(fontsize=11, fontweight="bold", color="white"))
    ax.legend(w, [f"{l}  ({v:.1f} s)" for l, v in zip(labs, vals)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=9, frameon=False)
    ax.set_title(titulo, fontweight="bold")
pie(a1, ser, cores_ser, "Serial (CPU)")
pie(a2, gpu, cores_gpu, "GPU (RTX 4070 SUPER)")
fig.suptitle("Perfil de execução: onde o tempo é gasto", fontweight="bold")
fig.text(0.5, 0.02, "Em ambas, a verificação de contatos do suscetível (S) é o gargalo — acesso aleatório à memória.",
         ha="center", fontsize=9.5, color="0.3")
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(OUT, "profile_serial_gpu.png")); plt.close(fig)
print("salvo: profile_serial_gpu.png")
