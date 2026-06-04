# Resultados da epidemia por cidade (GPU, R0=3.5)

Simulação completa da epidemia (GPU full-sim) para cada cidade, no **L real**, com o **Beta calibrado
da GPU** (R0=3.5, ver `r0_calibration_gpu.md`). **MAXSIM=50** (≥30 pela bimodalidade — ver
`bimodalidade.md`). DAYS=400, IPini=5, ocupação inicial leitos/UTI 50%.

Branch: `sim/resultados-cidades`. Hardware: RTX 4070 SUPER.

## Parâmetros e resultados

| Cidade | L | Beta (GPU) | Densidade | Contatos | Ataque (1−S) | Pico infecciosos (dia) | Pico H (dia) | Recuperados | Mortes COVID | Tempo |
|--------|------|-----------|-----------|----------|--------------|------------------------|--------------|-------------|--------------|-------|
| Rocinha | 264 | 0.00490 | Alta (Moore) | 2–120 | 58.8% | 0.63% (d175) | 0.03% (d260) | 58.6% | 10.2% | 136 s |
| Brasília | 1604 | 0.0995 | Baixa (Von Neumann) | 2 | 76.8% | 1.05% (d222) | 0.12% (d173) | 76.6% | 11.7% | 135 s |
| Manaus | 1343 | 0.0995 | Baixa (Von Neumann) | 2 | 75.9% | 1.05% (d217) | 0.09% (d160) | 75.8% | 12.1% | 142 s |
| **São Paulo** | 3355 | 0.0243 | Alta (Moore) | 2–19 | **72.6%** | **1.02% (d226)** | **0.11% (d193)** | **72.4%** | **11.0%** | **634 s** |

> ✅ **As 4 cidades concluídas com curvas completas** em `benchmarks/curvas/<cidade>/*.dat`
> (epidemicsprevalence, epidemicsincidence, Infectiousprevalence, Infectiousincidence — 400 dias).
> São Paulo só completou após corrigir **dois bugs de infraestrutura** (TDR do Windows + RNG
> degenerado que travava kernels) — ver `tdr_investigation.md`. O ataque da SP (72.6%) é
> consistente entre rodadas (MAXSIM=5 isolado deu 72.8%), confirmando que os guards anti-trava
> não enviesaram o resultado.

Colunas dos `.dat`: `dias S E IP IA TotalInfectious H ICU Recovered DeadCovid` (médias, proporções).
Dados brutos por cidade: `res_ROC.dat`, `res_BRA.dat`, `res_MAN.dat` (e `res_SP.dat`).

## Observações
- Todas calibradas para R0=3.5, mas as taxas de ataque diferem por cidade — efeito de **L** (tamanho
  finito), **densidade** (Moore 8 vs Von Neumann 4) e **estrutura de contatos**.
- **Rocinha** tem o menor ataque (58.8%) apesar de muitos contatos aleatórios (2–120): L pequeno
  (264) → mais efeito de tamanho finito / variância de bimodalidade.
- Brasília/Manaus (baixa densidade, ~2 contatos) saturam mais alto (~76–77%).
- Picos de infecciosos tardios (dia ~175–222): epidemias de propagação relativamente lenta no lattice.
- Mortes COVID 10–12% da população ao fim de 400 dias.
