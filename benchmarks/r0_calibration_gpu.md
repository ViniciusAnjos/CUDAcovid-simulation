# Calibração do Beta da GPU por cidade (R0 = 3.5)

Calibração **separada** do Beta da GPU (full-sim), contornando a diferença intrínseca de ~3% no
spread entre serial e GPU (ver `spread_investigation.md`). Cada implementação tem seu próprio Beta
para o mesmo R0=3.5 — abordagem pragmática alinhada ao plano da monografia.

## Metodologia

- **Medidor:** `r0_gpu.cu` — mesma definição do `r0_serial.cu` (1 paciente-zero, conta secundários
  `New_E` enquanto ele espalha IP/IA/ISLight; early-stop quando `IP+IA+ISLight != 1`), mas usando os
  **kernels reais do full-sim da GPU** (`S_kernel`, `IP_kernel`, `IS_kernel`, `spreadInfection`,
  `update_kernel`). Contadores lidos via `getCountersFromDevice` (d_New_E conta S→E).
- `IPini=1`, L=200 (R0 independente de L; GPU é rápida ~5 s/rodada), MAXSIM=1000.
- Bisseção por escala linear `Beta_novo = Beta·3.5/R0`. Cidade via `int city=...` no `r0_gpu.cu`.
- GPU usa densidade **runtime** (`getNeighborIndices` com `d_Density`): SP/ROC=Moore(8),
  BRA/MAN=Von Neumann(4) — já correto (o serial só passou a fazer isso após o fix do `#if`).

## Resultados

| Cidade | trajeto Beta→R0 | **Beta GPU (R0=3.5)** | R0 |
|--------|-----------------|-----------------------|-----|
| São Paulo | 0.0235→3.42 ; 0.0241→3.47 ; **0.0243→3.50** | **0.0243** | 3.50 |
| Rocinha | 0.0051→3.65 ; **0.004896→3.54** | **0.00490** | 3.54 |
| Brasília | 0.112→3.81 ; 0.103→3.59 ; **0.0995→3.50** | **0.0995** | 3.50 |
| Manaus | (= Brasília) | **0.0995** | 3.50 |

## Comparação Serial × GPU (Beta para R0=3.5)

| Cidade | Serial | GPU | GPU/Serial | Vizinhança |
|--------|--------|-----|-----------|------------|
| São Paulo | 0.02129 | **0.0243** | 1.14 | Moore (8) |
| Rocinha | 0.00461 | **0.00490** | 1.06 | Moore (8) |
| Brasília | 0.102 | **0.0995** | 0.98 | Von Neumann (4) |
| Manaus | 0.102 | **0.0995** | 0.98 | Von Neumann (4) |

**Por que diferem (e por que isso resolve o problema do spread):**
- No mesmo Beta, o R0 da GPU é **menor** que o do serial (ex.: SP Beta=0.02129 → serial 3.52, GPU 3.18).
  Causa: a GPU tem ~12% **menos** agente-dias infecciosos (medido no `spread_investigation.md` Exp 9),
  efeito que no R0 (1 geração) domina o ~3% a mais por passo. Por isso a GPU precisa de Beta **maior**
  para SP/ROC (Moore, muitos contatos).
- Em BRA/MAN (Von Neumann, ~2 contatos) os dois efeitos quase se cancelam → GPU ≈ serial (−2%).
- **Calibrar cada implementação com seu próprio Beta faz ambas baterem R0=3.5**, contornando a
  diferença intrínseca de ~3% no mecanismo (em vez de tentar eliminá-la no código).

## Como reproduzir
```
# worktree CudaRuntime1-gpu/CudaRuntime1, define.h: IPini=1, L=200, MAXSIM=1000, Beta=<tabela>
# r0_gpu.cu: int city = <CIDADE>;
nvcc r0_gpu.cu -o r0_gpu.exe -arch=sm_89 --diag-suppress 20091 --diag-suppress 177
.\r0_gpu.exe   # imprime "R0 medio = ..."
```
