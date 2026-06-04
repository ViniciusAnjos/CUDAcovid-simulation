# Resumo dos resultados — Paralelização CUDA da simulação COVID (ABM)

Monografia UFF — Vinícius Santos Anjos da Silva. Índice e síntese de toda a investigação, validação,
benchmark de desempenho e análise estatística. Cada seção aponta para os documentos e figuras de
detalhe (em `benchmarks/` e `benchmarks/curvas/graficos/`).

GPU de referência: **NVIDIA RTX 4070 SUPER** (Ada, sm_89, 7168 cores, 48 MB L2, 504 GB/s, 12 GB).

---

## 1. Objetivo
Paralelizar em CUDA o ABM SEIR-estendido de COVID-19 em 4 cidades brasileiras (São Paulo, Rocinha,
Brasília, Manaus) e **validar que a GPU reproduz fielmente o serial de referência**, medindo o
ganho de desempenho.

## 2. Correções (o caminho até a equivalência)

### Infraestrutura da GPU (rodadas grandes travavam)
- **WDDM TDR (2 s):** a GPU de display resetava o driver quando um kernel de pico passava de 2 s.
  Provado pelos eventos `nvlddmkm 153`. Fix: `TdrDelay=60` + checagem de erro CUDA no `covid.cu`.
- **RNG degenerado:** LCG quebrado (multiplicador 888121 ≡ 1 mod 8, semente par em sim par) colapsava
  em ciclo curto → laços de rejeição `do…while` em loop infinito. Fix: *guards* (limite + fallback).
- Otimização **Health-SoA**: campo `Health` num array compacto (1 B/célula) que cabe no L2 →
  **~12–16×** em São Paulo (acesso aleatório vira cache hit). Detalhes: `tdr_investigation.md`,
  `otimizacao_health_soa.md`.

### Equivalência serial ↔ GPU (2 bugs no SERIAL; a GPU estava correta)
Com o mesmo β, o serial divergia da GPU. Bisseção empírica (descartando RNG, duração infecciosa e
parâmetros por medição) isolou **dois bugs no serial**:
1. **`Update.h` — isolamento ligado por engano:** `#if(BeginOfIsolation==ON)` avaliado pelo
   pré-processador (`0==0` sempre verdadeiro) isolava 50% da população → ~92% da divergência.
   Fix: `if` runtime.
2. **`S.h` — Sfunc apagava infecções:** `else Swap=S` sobrescrevia infecção do infected-driven.
   Fix: `else if (Checked==0) Swap=S`. (Resíduo ~2–3%.)
Detalhes: `equivalencia_state_machine.md`.

## 3. Validação — mesmo β → mesmas curvas (4 cidades)
Após os fixes, serial e GPU produzem a **mesma epidemia no mesmo β** (L real, MAXSIM=50):

| Cidade | β (R0=3,5) | Ataque GPU / Serial | RMSE de S(t) |
|--------|-----------|---------------------|--------------|
| Rocinha | 0,0049 | 58,7% / 58,8% | 0,76 p.p. |
| Manaus | 0,0995 | 75,9% / 75,4% | 0,86 p.p. |
| Brasília | 0,0995 | 76,8% / 76,3% | 1,13 p.p. |
| São Paulo | 0,0243 | 72,6% / 72,8% | 0,74 p.p. |

RMSE < 1,2 p.p. (infecciosos < 0,03 p.p.) = **ruído estatístico** → equivalência confirmada.
Figuras: `graficos/validacao/` (overlays serial×GPU + `equivalencia_rmse.png`).
**Não há necessidade de β separado por implementação** — a calibração separada anterior só
compensava os bugs do serial.

## 4. Resultados epidemiológicos
Curvas completas (400 dias) por cidade — prevalência, incidência, carga clínica (Hospital/UTI),
mortes. Figuras de apresentação em `graficos/resultados/` (serial e comparação serial×GPU).
Curvas brutas em `curvas/<cidade>/` e `curvas/validacao/`.

## 5. Benchmark de desempenho (speedup)
Mesma epidemia (mesmo β), MAXSIM=50, serial CPU `-O3 /O2` × GPU:

| Cidade | L | Células | Serial | GPU | **Speedup** |
|--------|------|---------|--------|-----|-------------|
| Rocinha | 264 | 70 k | 235 s | 129 s | **1,8×** |
| Manaus | 1343 | 1,80 M | 1 891 s | 136 s | **13,9×** |
| Brasília | 1604 | 2,57 M | 2 124 s | 131 s | **16,2×** |
| São Paulo | 3355 | 11,25 M | 35 755 s (9,9 h) | 634 s | **56,4×** |

O speedup **cresce com o tamanho do grid** (de ~2× a ~56×). Figuras: `graficos/speedup/`.

## 6. Análise de hardware (RTX 4070 SUPER)
- **Memory-bound:** o block size é irrelevante (64→1024 ≈ mesmo tempo); o gargalo é o acesso
  aleatório à memória.
- **Profile:** o `S_kernel` (contatos do suscetível) domina — **60% GPU / 88% serial**.
- **Memória/cache:** o array `GPUPerson` (64 B/cél.) estoura o L2 em L≈866; L máx na 4070S ≈ 13 700.
- **Escala × L:** a GPU tem piso de overhead ~2,5 s/sim (WDDM); speedup 1× (L=200) → 53× (L=3200).
- Figuras: `graficos/hardware/` (`memoria_vs_L`, `health_soa`, `escala_L`, `blocksize`,
  `profile_serial_gpu`).

## 7. Comparação entre GPUs (4070 SUPER × GTX 1050 Ti)
Mesmo código/β nas duas placas → **ataque idêntico** (independência de hardware) e tempo da 1050 Ti
**4–6×** maior nas cidades grandes. Essa razão bate com a de **banda de memória (4,5×)** e **não**
com a de cálculo FP32 (17×) → **confirma memory-bound por outra via**.
Doc: `comparacao_gpus.md` · figura: `graficos/hardware/comparacao_gpus.png`.

## 8. Análise estatística (variabilidade)
50 simulações por cidade: no R0=3,5 o desvio do ataque é **≤ 0,9 p.p.** (resultados robustos, sem
bimodalidade); a incerteza cresce **~7×** perto do limiar epidêmico (extinção estocástica).
Doc: `variabilidade.md` · figuras: `graficos/variabilidade/`.

---

## Onde está cada coisa (GitHub: `ViniciusAnjos/CUDAcovid-simulation`)
- **Branch `r0`** — serial de referência com os 2 fixes (`Update.h`, `S.h`) + scripts de benchmark.
- **Branch `fix/state-machine-timing`** — investigação completa, validação, todos os benchmarks,
  gráficos e docs (incluindo este resumo).

### Documentos de detalhe (`benchmarks/`)
`tdr_investigation.md` · `otimizacao_health_soa.md` · `equivalencia_state_machine.md` ·
`benchmark_serial_gpu.md` · `comparacao_gpus.md` · `variabilidade.md` ·
`epidemia_cidades.md` · `README_benchmark_gpu.md` (rodar em outra GPU).

### Scripts de plotagem (`benchmarks/curvas/`)
`plot_resultados.py` · `plot_comparacao.py` · `plot_speedup.py` · `plot_hardware.py` ·
`plot_profile.py` · `plot_gpus.py` · `plot_variabilidade.py` · `equivalencia_quant.py`.
