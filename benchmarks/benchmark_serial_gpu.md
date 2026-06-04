# Benchmark Serial × GPU — curvas e tempo por cidade

Comparação da implementação **serial (CPU, 1 thread)** com a **paralela (GPU, RTX 4070 SUPER)**,
cidade por cidade, no **L real** e **MAXSIM=50** (400 dias). Objetivo duplo:
1. **Validação das curvas** — as duas implementações produzem a mesma epidemia?
2. **Benchmark de otimização** — speedup da paralelização (tempo serial / tempo GPU).

## Metodologia

- **Compilação com otimização** nos dois lados:
  - Serial: `nvcc kernel.cu -O3 -Xcompiler "/O2 /wd4716"`.
  - GPU: `nvcc covid.cu -arch=sm_89` (+ otimização de memória Health-SoA, ver `otimizacao_health_soa.md`).
- **Tempo** medido como wall-clock do executável (mesma métrica nos dois; mesma máquina/condições,
  rodadas back-to-back).
- **Calibração de β no full-sim (decisão do autor):** calibrar pelo R0 paciente-zero **não** faz as
  epidemias completas baterem (ver `r0_calibration_gpu.md`). Então, para cada cidade, calibra-se o
  **β do serial** até o **ataque final** casar com o da GPU. Assim as duas rodam a **mesma epidemia**
  (mesma taxa de ataque) → o benchmark de tempo compara cargas equivalentes.
- **β da GPU:** o calibrado para R0=3.5 da GPU (`r0_calibration_gpu.md`).

> ⚠️ **Achado importante:** casar o **ataque** via β **não casa a forma temporal** da curva. Com β
> mais alto, o serial fica mais "rápido e pontudo" (pico antes e mais alto); a GPU, com β menor, é
> mais lenta e achatada. Isso reflete a diferença de ~12% na duração infecciosa entre as
> implementações (tempo de geração diferente). Equivalência **exata** das curvas exigiria alinhar o
> state-machine (TimeOnState/StateTime) — fora do escopo deste benchmark.

---

## Rocinha (L=264) — ✅ concluída

Cidade de **menor porte** (264×264 = 69 696 células). Alta densidade (Moore 8), contatos 2–120.

### Calibração do β serial (full-sim, alvo ataque GPU = 58,7%)
Sweeps MAXSIM=20 (β → ataque):

| β | 0.0050 | 0.0062 | 0.0070 | 0.0080 | 0.0090 | 0.0105 | 0.0120 | 0.0140 | 0.0160 |
|------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ataque | 4,1% | 14,4% | 23,1% | 32,8% | 40,5% | 48,4% | 54,1% | 59,6% | 63,5% |

→ interpolado **β_serial = 0,0137** (confirmado MAXSIM=50: ataque 58,78%).

### Resultado (MAXSIM=50, 400 dias)
| | β | Ataque | Recuperados | Mortes COVID | **Tempo** | s/sim |
|---|------|--------|-------------|--------------|-----------|-------|
| Serial (CPU /O2) | 0,0137 | 58,78% | 58,8% | 10,4% | **142,4 s** | 2,85 |
| GPU (RTX 4070S) | 0,0049 | 58,73% | 58,6% | 10,1% | **128,6 s** | 2,57 |
| **Speedup** | | (casado) | | | **1,11×** | |

**Leitura:** para a Rocinha a GPU **mal supera** o serial (1,11×). O grid é minúsculo (~70 mil
células) → a GPU fica **subutilizada** (milhares de núcleos ociosos) e **dominada pelo overhead de
lançamento** (~10 kernels × 400 dias × 50 sims = 200 mil lançamentos+syncs). O ganho da
paralelização **cresce com o tamanho do problema** — esperado que SP (L=3355, ~161× mais células)
mostre speedup de ordens de magnitude.

**Curvas:** ataque final idêntico (≈58,7%), mas o serial (β alto) tem pico de infecciosos antes e
mais alto (d~100, 1,1%) que a GPU (d~160, 0,6%) — ver `graficos/ROC_comparacao_serial_gpu.png`.
Curvas serial em `curvas/serial/ROC/`, GPU em `curvas/ROC/`.

---

## Brasília (L=1604) — ✅ concluída

Baixa densidade (Von Neumann 4), ~2 contatos aleatórios. 1604×1604 = 2 572 816 células.

### Calibração do β serial (full-sim, alvo ataque GPU = 76,75%)
| β | 0.105 | 0.120 | 0.135 | 0.150 |
|------|-------|-------|-------|-------|
| ataque (MAXSIM=10) | 44,5% | 63,9% | 74,0% | 79,5% |

→ interpolado **β_serial = 0,142** (confirmado MAXSIM=50: ataque 76,86%).

### Resultado (MAXSIM=50, 400 dias)
| | β | Ataque | Mortes COVID | **Tempo** | s/sim |
|---|------|--------|--------------|-----------|-------|
| Serial (CPU /O2) | 0,142 | 76,86% | 11,2% | **2124 s** (35,4 min) | 42,5 |
| GPU (RTX 4070S) | 0,0995 | 76,75% | 11,7% | **131,3 s** | 2,63 |
| **Speedup** | | (casado) | | **16,2×** | |

**Leitura:** com L=1604 (2,6 M células, ~37× a Rocinha) a paralelização **compensa** — 16,2× contra
1,1× da ROC. As curvas serial×GPU ficam **bem próximas** (ataque e timing): BRA está saturada (~77%,
longe do limiar) → menos sensível ao β, pico de infecciosos quase coincidente (d~220). Ver
`graficos/BRA_comparacao_serial_gpu.png`. Curvas em `curvas/serial/BRA/` e `curvas/BRA/`.

## Manaus (L=1343) — ✅ concluída

Baixa densidade (Von Neumann 4), ~2 contatos. 1343×1343 = 1 803 649 células. Muito similar a BRA.

### Calibração do β serial (full-sim, alvo ataque GPU = 75,91%)
Probe β=0,140 (MAXSIM=10) → 75,21% → ajuste **β_serial = 0,142** (= BRA; confirmado MAXSIM=50:
ataque 76,05%).

### Resultado (MAXSIM=50, 400 dias)
| | β | Ataque | Mortes COVID | **Tempo** | s/sim |
|---|------|--------|--------------|-----------|-------|
| Serial (CPU /O2) | 0,142 | 76,05% | 11,8% | **~1535 s**¹ | ~30,7 |
| GPU (RTX 4070S) | 0,0995 | 75,91% | 12,1% | **135,7 s** | 2,71 |
| **Speedup** | | (casado) | | **~11,3×** | |

¹ Tempo derivado do ritmo medido (probe 30,7 s/sim × 50; consistente com o escalonamento de BRA por
L²). O cronômetro do job final perdeu o registro (wrapper sem saída), mas os dados das 50 sims estão
completos e válidos.

Curvas serial×GPU próximas (saturada ~76%), ver `graficos/MAN_comparacao_serial_gpu.png`.

## Diferença entre os resultados (Serial × GPU)

Com o β calibrado no full-sim para casar o **ataque**, comparamos as curvas de prevalência das duas
implementações (`graficos/<cidade>_prevalencia_serial_vs_gpu.png` — overlay + painel de diferença).

| Cidade | Ataque GPU/Ser (Δp.p.) | Mortes GPU/Ser (Δp.p.) | Pico infecciosos GPU / Ser (Δdia) | RMSE de S(t) | Máx \|ΔS\| |
|--------|------------------------|------------------------|-----------------------------------|--------------|-----------|
| Rocinha | 58,7% / 58,8% (**−0,0**) | 10,08% / 10,44% (−0,36) | 0,63%@d169 / 1,17%@d97 (**+72 d**) | **18,4 p.p.** | **+42,2 p.p.**@d113 |
| Brasília | 76,8% / 76,9% (**−0,1**) | 11,65% / 11,15% (+0,50) | 1,05%@d222 / 0,93%@d238 (−16 d) | 5,5 p.p. | −13,8 p.p.@d219 |
| Manaus | 75,9% / 76,0% (**−0,1**) | 12,13% / 11,75% (+0,38) | 1,05%@d216 / 0,91%@d235 (−19 d) | 6,0 p.p. | −15,0 p.p.@d216 |

(p.p. = pontos percentuais da população; RMSE/Máx ao longo de 400 dias, alinhados por dia.)

**Interpretação:**
- **O resultado final converge muito bem:** ataque idêntico a <0,1 p.p. (por construção da
  calibração) e mortes dentro de ±0,5 p.p. → as duas implementações produzem o **mesmo desfecho
  epidemiológico**.
- **A diferença está na dinâmica temporal (forma da curva), não no desfecho.** Quanto mais perto do
  limiar, maior a divergência de trajetória:
  - **Rocinha** (quase-crítica, β_serial 2,8× o da GPU): serial pica **72 dias antes** e ~2× mais
    alto; RMSE de S(t) = 18,4 p.p., chegando a 42 p.p. de diferença instantânea no meio da epidemia.
  - **Brasília/Manaus** (saturadas, β_serial só 1,43× o da GPU): curvas **bem próximas** (RMSE ~5–6
    p.p.); aqui a GPU pica **antes** (−16 a −19 d) — o tempo de geração mais curto da GPU domina
    quando a razão de β é menor.
- **Causa** (já documentada em `r0_calibration_gpu.md`): casar o ataque via β não casa o **tempo de
  geração** (a GPU tem ~12% menos agente-dias infecciosos). Equivalência **exata** das curvas
  exigiria alinhar o state-machine (TimeOnState/StateTime) — fora do escopo deste benchmark.

## São Paulo (L=3355) — pendente
> ⚠️ SP serial é o caso caro (~100 min por rodada MAXSIM=50; sweep de calibração custa horas).
> Estratégia a definir (calibração com MAXSIM reduzido / poucos pontos).

---

## Tabela-resumo de speedup (em construção)

| Cidade | L | Células | Tempo Serial | Tempo GPU | **Speedup** |
|--------|------|---------|--------------|-----------|-------------|
| Rocinha | 264 | 69 696 | 142,4 s | 128,6 s | **1,11×** |
| Brasília | 1604 | 2,57 M | 2124 s | 131,3 s | **16,2×** |
| Manaus | 1343 | 1,80 M | ~1535 s | 135,7 s | **~11,3×** |
| São Paulo | 3355 | 11,25 M | — | — | — |
