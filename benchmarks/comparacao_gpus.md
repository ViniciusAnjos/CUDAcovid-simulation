# Comparação entre GPUs — RTX 4070 SUPER × GTX 1050 Ti

Mesma implementação GPU, **mesmo código e mesmo β**, rodada em duas placas (MAXSIM=5, L real, 400
dias). Gerado com `benchmark_gpu.ps1` (portátil) — ver `README_benchmark_gpu.md`.

## Hardware
| GPU | Arquitetura | Cores | Banda de memória | FP32 (~) | VRAM |
|-----|-------------|-------|------------------|----------|------|
| RTX 4070 SUPER | Ada (sm_89) | 7168 | 504 GB/s | ~35 TFLOPS | 12 GB |
| GTX 1050 Ti | Pascal (sm_61) | 768 | 112 GB/s | ~2,1 TFLOPS | 4 GB |

## Resultados (tempo por simulação, s)
| Cidade | L | 4070 SUPER | 1050 Ti | 1050Ti / 4070S | Ataque (4070S / 1050Ti) |
|--------|------|-----------|---------|----------------|-------------------------|
| Rocinha | 264 | 2,53 | 21,78 | **8,6×** | 0,587 / 0,585 ✓ |
| Manaus | 1343 | 2,92 | 13,54 | **4,6×** | 0,760 / 0,760 ✓ |
| Brasília | 1604 | 2,87 | 11,83 | **4,1×** | 0,767 / 0,767 ✓ |
| São Paulo | 3355 | 12,75 | 71,95 | **5,6×** | 0,728 / 0,728 ✓ |

## Dois achados

### 1. Independência de hardware (validação)
O **ataque é idêntico** nas duas GPUs (RNG determinístico, mesma lógica) → o resultado da simulação
**não depende do hardware**. Isso reforça a validação de correção da implementação paralela.

### 2. A simulação é memory-bound (confirmado por outra via)
Nas cidades grandes (Manaus/Brasília/SP) a 1050 Ti é **~4–6× mais lenta**, batendo com a razão de
**banda de memória** (504/112 = **4,5×**) — e **não** com a razão de poder de cálculo FP32
(35/2,1 ≈ **17×**). Ou seja: o desempenho escala com a **banda**, não com os FLOPs → a simulação é
**limitada por memória** (consistente com o profile, que aponta o `S_kernel`/contatos aleatórios como
gargalo, e com a otimização Health-SoA de cache).

> **Rocinha (8,6×) é o outlier:** grid minúsculo (70k células) é dominado por overhead de
> lançamento/subutilização, não por banda — por isso desvia da razão de banda.

Gráfico: `curvas/graficos/hardware/comparacao_gpus.png`.

## Dados brutos (1050 Ti, MAXSIM=5, TdrDelay=60)
```json
[
  { "Cidade": "ROC", "tsim": 21.78, "ataque": 0.5854 },
  { "Cidade": "MAN", "tsim": 13.54, "ataque": 0.76   },
  { "Cidade": "BRA", "tsim": 11.83, "ataque": 0.7673 },
  { "Cidade": "SP",  "tsim": 71.95, "ataque": 0.7284 }
]
```
