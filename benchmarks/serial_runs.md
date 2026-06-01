# Benchmark & Sumário de Simulações

Registro de cada rodada da simulação (serial e GPU): parâmetros, tempo de parede (benchmark) e
sumário dos resultados. Atualizado automaticamente a cada execução.

Colunas dos `.dat`: `dias S E IP IA TotalInfectious H ICU Recovered DeadCovid` (médias, proporções 0–1).

---

## Run #1 — Serial / São Paulo

| Campo | Valor |
|-------|-------|
| Data | 2026-05-31 |
| Branch | `r0` |
| Binário | `serial_sim.exe` (kernel.cu) |
| Cidade | São Paulo (`cities(SP)`) |
| Densidade | Alta (Moore, 8 vizinhos) |
| L | 3355 (~11,26M agentes) |
| MAXSIM | 10 |
| DAYS | 400 |
| Beta | 0.0658 (calibrado R0=3.5 serial) |
| Leitos/Pop | 0.00247452 |
| UTI/Pop | 0.00043782 |
| Contatos aleatórios | 1.5–2.5 (preset do código) |
| Ocupação inicial leitos/UTI | 50% |

> ⚠️ **Discrepância documentada:** a tabela do CLAUDE.md lista os contatos aleatórios de SP como
> "2–19", mas o preset de São Paulo em `cities.h` usa 1.5–2.5. Mantive o preset do código porque o
> `Beta = 0.0658` (R0=3.5) foi calibrado para essa estrutura de contatos; trocar para 2–19 sem
> recalibrar o Beta inflaria muito o R0. A decidir se recalibra para os contatos 2–19.

**Tempo de execução (benchmark):**

| Métrica | Valor |
|---------|-------|
| Tempo total (10 sims) | **~98 min** (1h38) |
| Tempo médio por simulação | **~9.8 min/sim** |
| Janela | fim sim1 23:53:04 → fim sim10 01:20:58 |
| Hardware | CPU (serial, single-thread) |

> Tempo reconstruído pelos timestamps dos `prevalence_N.dat` (o cronômetro do wrapper não gravou
> `bench_time_run1.txt`). Cadência consistente: 9.5–10.1 min/sim.

**Sumário dos resultados** (médias sobre 10 sims, proporções da população):

| Métrica | Valor | Dia |
|---------|-------|-----|
| Pico de TotalInfectious | 0.0056 (0.56%) | 372 |
| Pico de E (expostos) | 0.0680 (6.80%) | 366 |
| Pico de H (hospital) | 0.0012 (0.12%) | 335 |
| Pico de ICU (UTI) | 0.0001 (0.01%) | 399 |
| Mínimo de S | 0.4798 | 400 |
| **Estado final (dia 400)** | | |
| S final | 0.4798 (47.98%) | |
| Recuperados | 0.4449 (44.49%) | |
| Mortes COVID | 0.0615 (6.15%) | |
| **Taxa de ataque** (1 − S_final) | **0.5202 (52.0%)** | |

**Observações:**
- Epidemia de **propagação lenta**: pico de infectados só no dia ~372. Coerente com lattice grande
  (3355²) e espalhamento predominantemente local (Moore + contatos aleatórios baixos 1.5–2.5),
  formando uma onda que leva muitos dias para atravessar a grade.
- Diferente do regime L=100 (saturação rápida, S→~0.06 por volta do dia 25): com L grande a dinâmica
  é dominada pela frente de onda, não por mistura global.
- Taxa de ataque ~52% e mortalidade COVID ~6.15% da população ao fim de 400 dias.

---

## Run #2 — GPU / São Paulo

| Campo | Valor |
|-------|-------|
| Data | 2026-06-01 |
| Branch | `fix/gpu-kernel-correctness` (worktree `CudaRuntime1-gpu`) |
| Binário | `covid_sim.exe` (covid.cu) |
| Hardware | NVIDIA RTX 4070 SUPER (sm_89) |
| Cidade | São Paulo (`city = SP`) |
| Densidade | Alta (Moore) |
| L | 3355 (11.256.025 células) |
| MAXSIM | 10 |
| DAYS | 400 |
| Beta | 0.0658 (mesmo do serial) |
| Leitos / UTI | 27735 leitos, 4928 UTI (50% disponíveis) |

> 🐛 **Bug encontrado e corrigido no worktree:** `covid.cu` tinha `const int L = 100` e
> `const int MAXSIM = 5` **hardcoded dentro do `main()`** ("small for testing"), fazendo *shadow*
> das globais do `define.h` — por isso a GPU sempre rodava L=100 independentemente do `define.h`.
> Removidas as declarações locais para usar L/N/MAXSIM globais. (Correção ainda não commitada.)

**Tempo de execução (benchmark):**

| Métrica | Valor |
|---------|-------|
| Tempo total (10 sims) | **67.5 s** |
| Tempo médio por simulação | ~6.75 s/sim |

**Sumário dos resultados** (médias de 10 sims):

| Métrica | Valor | Dia |
|---------|-------|-----|
| Pico de TotalInfectious | 0.0118 (1.18%) | 252 |
| Pico de E (expostos) | 0.1434 (14.34%) | 246 |
| Pico de H (hospital) | 0.0012 (0.12%) | 187 |
| Pico de ICU | 0.0001 (0.01%) | 345 |
| S final | 0.1886 (18.86%) | 400 |
| Recuperados (final) | 0.8095 (80.95%) | 400 |
| Mortes COVID (final) | 0.1268 (12.68%) | 400 |
| **Taxa de ataque** (1 − S_final) | **0.8114 (81.1%)** | 400 |

---

## Comparação Serial × GPU (São Paulo, L=3355, MAXSIM=10, Beta=0.0658, 400 dias)

### ⏱️ Benchmark / Speedup

| Implementação | Tempo total | Por sim | Speedup |
|---------------|-------------|---------|---------|
| Serial (CPU, 1 thread) | ~97.7 min (5862 s) | ~9.8 min | 1× |
| GPU (RTX 4070 SUPER) | 67.5 s | ~6.75 s | **~86.9×** |

### 📊 Resultados (dia 400)

| Métrica | Serial | GPU | Δ |
|---------|--------|-----|---|
| Taxa de ataque (1 − S) | 52.0% | 81.1% | +29 pp |
| Recuperados | 44.5% | 81.0% | +36 pp |
| Mortes COVID | 6.15% | 12.68% | +6.5 pp |
| Pico TotalInfectious | 0.56% (dia 372) | 1.18% (dia 252) | maior e mais cedo |
| Pico E | 6.80% (dia 366) | 14.34% (dia 246) | maior e mais cedo |

> ⚠️ **Os resultados NÃO batem (divergência conhecida).** Com o mesmo L, Beta e nº de sims, a GPU
> produz uma epidemia **mais intensa e mais rápida** (pico ~120 dias antes, taxa de ataque 81% vs
> 52%). Isto é o problema de correção em aberto descrito no CLAUDE.md (mecanismo
> `spreadInfection_kernel` vs `S_kernel`, uso de `Checked`/`Exponent`). **O speedup (~87×) é
> confiável; a equivalência estatística serial↔GPU ainda precisa ser corrigida/calibrada.**

---

## Run #3 — GPU / São Paulo com Beta=0.0163 (calibração R0 do branch r0)

Mesmos parâmetros da Run #2, **exceto Beta = 0.0163** (valor que deu R0=3.5 no modo paciente-zero
do branch `r0`).

| Métrica | Valor |
|---------|-------|
| Tempo total (10 sims) | 69.8 s |
| Pico de E | **0.000000** (dia 267) |
| Pico de TotalInfectious | **0.000000** (dia 267) |
| Máx. Recuperados | 0.000003 |
| S final | **1.0000** |
| Mortes COVID | 0.0000 |
| Taxa de ataque | **~0% (epidemia extinta)** |

### 🔬 Achado crítico: a calibração R0 do branch `r0` NÃO transfere para o full-sim

| Beta | R0 (paciente-zero, r0) | Resultado no full-sim GPU |
|------|------------------------|---------------------------|
| 0.0163 | 3.5 (medido) | **epidemia extinta** (R0_efetivo < 1) |
| 0.0658 | — | epidemia explode (ataque 81%) |

- Se `Beta=0.0163` realmente desse R0=3.5 (>1) no full-sim, a epidemia **deveria** decolar — mas
  se extingue. Logo, **a transmissão secundária no full-sim se comporta diferente da primária**
  medida no modo paciente-zero.
- Consistente com a hipótese do CLAUDE.md: no full-sim, o `spreadInfection_kernel` dos casos
  secundários pode estar sendo bloqueado (ordem de kernels / `Checked`=1 do `S_kernel` no mesmo
  passo), de modo que cada caso infecta muito menos que os 3.5 medidos no paciente-zero.
- **Conclusão:** o full-sim GPU precisa de **calibração própria** (variar Beta no full-sim até
  bater R0≈3.5 / casar com o serial), OU corrigir o bug de transmissão secundária antes. A
  calibração paciente-zero (0.0163) não é representativa.

> O **speedup (~70 s, ~84×) independe do Beta** — o tempo é dominado pelo tamanho do grid.

---
