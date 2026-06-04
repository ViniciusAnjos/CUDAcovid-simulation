# CUDAcovid-simulation — CLAUDE.md

## Visão geral do projeto

Monografia de graduação (UFF - Campus Volta Redonda) de Vinícius Santos Anjos da Silva.
Orientador: Aquino Lauri de Espíndola.

**Objetivo:** Paralelizar em CUDA uma simulação epidemiológica baseada em agentes (ABM) de COVID-19
em áreas urbanas do Brasil, validando que os resultados GPU convergem para os mesmos resultados
da implementação serial original.

---

## Modelo epidemiológico

Modelo compartimental SEIR estendido. Estados de saúde:

```
S(suscetível) → E(exposto) → IP(pré-sintomático) → IA(assintomático) → R
                                                  → ISLight → R ou ISModerate
                                                  → ISModerate → R ou ISSevere
                                                  → ISSevere → H(hospital) → R ou ICU
                                                                           → ICU → R ou DeadCovid
```

Mortes naturais ocorrem em qualquer estado → pessoa é substituída por novo S.

### Mecanismo de contágio (CRÍTICO)

A transmissão ocorre por **dois mecanismos simultâneos** — ambos devem existir na GPU:

1. **Suscetível procura infectados** (`Neighbors.h` / `S_kernel`):
   - Verifica vizinhos locais (Moore=8 vizinhos para alta densidade, Von Neumann=4 para baixa)
   - Verifica contatos aleatórios
   - Calcula `P = 1 - (1-β)^n` e decide infecção

2. **Infectado procura suscetíveis** (`Neighborsinfected.h` / `spreadInfection_kernel`):
   - Chamado por: IP, IA e ISLight **enquanto ainda estão no período infeccioso** (else do TimeOnState)
   - ISModerate e ISSevere **NÃO chamam** Neighborsinfected
   - Usa `Exponent` e `Checked` para evitar dupla infecção no mesmo passo

### Parâmetro Beta

- `β = 0.0658` calibrado para `R0 = 3.5` na simulação serial
- Beta é parâmetro da doença (igual para todas as cidades)
- **Cada implementação (serial/GPU) precisa ser calibrada separadamente** para confirmar que
  o mesmo β produz R0 = 3.5, pois diferenças no mecanismo de contato podem alterar o R0 efetivo
- Calibração: medir média de infectados a partir de 1 caso, variar β até R0 ≈ 3.5

---

## Parâmetros por cidade

| Cidade    | Densidade | L    | Leitos/Pop  | UTI/Pop     | Contatos aleatórios |
|-----------|-----------|------|-------------|-------------|---------------------|
| São Paulo | Alta      | 3355 | 0.00247452  | 0.00043782  | 2–19                |
| Rocinha   | Alta      | 264  | 0.00055111  | 0.00014592  | 2–120               |
| Brasília  | Baixa     | 1604 | 0.00260879  | 0.00040114  | 2                   |
| Manaus    | Baixa     | 1343 | 0.00187124  | 0.00027858  | 2                   |

Alta densidade → vizinhança de Moore (8 vizinhos).
Baixa densidade → vizinhança de Von Neumann (4 vizinhos).

---

## Parâmetros da doença

| Parâmetro      | Duração (dias) | Probabilidades de transição |
|----------------|----------------|-----------------------------|
| Latência (E)   | 0–27           | —                           |
| Pré-sint. (IP) | 0–14           | P(IP→IA)=0.5                |
| Assint. (IA)   | 0–7            | P(IP→ISLight)=0.6, P(IP→ISModerate)=0.2, P(IP→ISSevere)=0.2 |
| ISLight        | 0–14           | P(ISLight→ISModerate)=0.1   |
| ISModerate     | 0–28           | Depende de ProbRecoveryModerate[idade] |
| ISSevere       | 0–4            | P(ISSevere→H)=0.75 se cama disponível |
| Hospital (H)   | 7–45           | Depende de ProbRecoveryH[idade]        |
| UTI (ICU)      | 10–60          | Depende de ProbRecoveryICU[idade]      |

Ocupação inicial dos leitos: 50% já ocupados por outras doenças.
Início: 5 indivíduos em IP, restante S.

---

## Estrutura de arquivos

### Serial (referência)
| Arquivo | Papel |
|---------|-------|
| `kernel.cu` | Main serial — loop principal |
| `define.h` | Constantes globais (estados, parâmetros, L, MAXSIM) |
| `S.h` | Função Sfunc — chama Neighborsfunc |
| `E.h` | Função Efunc |
| `IP.h` | Função IPfunc — chama Neighborsinfectedfunc quando ainda infeccioso |
| `IS.h` | ISfunc — chama Neighborsinfectedfunc para IA e ISLight |
| `H.h` | Hfunc |
| `ICU.h` | ICUfunc |
| `Neighbors.h` | Suscetível verifica vizinhos + contatos aleatórios |
| `Neighborsinfected.h` | Infectado tenta infectar suscetíveis aleatórios |
| `Update.h` | Updatefunc — atualiza lattice, conta estados, substitui mortos |
| `begin.h` | Inicialização da população |
| `cities.h` | Parâmetros por cidade |
| `death.h` | ProbNaturalDeath |
| `agestructure.h` | Estrutura etária brasileira |
| `probsrecovery.h` | Arrays de recuperação por idade |
| `isolation.h` / `isolation_correct.h` | Políticas de isolamento |

### GPU (implementação paralela)
| Arquivo | Papel |
|---------|-------|
| `covid.cu` | Main GPU — orquestra kernels e saída |
| `gpu_define.cuh` | Constantes CUDA (`__constant__`), setup da GPU, buildArrays |
| `gpu_person.cuh` | Struct `GPUPerson` |
| `gpu_utils.cuh` | `to1D`, `to2D`, `getNeighborIndices`, `wrapIndex` |
| `gpu_aleat.cuh` | RNG per-thread (LCG simples) |
| `gpu_neighbors.cuh` | `checkAllContacts`, `checkLocalContacts`, `checkRandomContacts`, `spreadInfection_kernel` |
| `gpu_begin.cuh` | `initPopulation_kernel`, `distributeInitialInfections_kernel` |
| `gpu_update_boundaries.cuh` | Condições de contorno periódicas |
| `S_kernel.cuh` | Kernel estado S |
| `E_kernel.cuh` | Kernel estado E |
| `IP_kernel.cuh` | Kernel estado IP — chama `spreadInfection_kernel` |
| `IS_kernel.cuh` | Kernel estados IA/ISLight/ISModerate/ISSevere |
| `H_kernel.cuh` | Kernel estado H |
| `ICU_kernel.cuh` | Kernel estado ICU |
| `Update_kernel.cuh` | Kernel de atualização + contadores de prevalência/incidência |
| `output_files.cuh` | Escrita dos arquivos .dat de saída |

### Testes
| Arquivo | Papel |
|---------|-------|
| `test_runner.cu` | Entry point dos testes unitários |
| `test_update_kernel.cuh` | 6 testes do Update_kernel (todos passando) |

---

## Bugs já corrigidos (branch `fix/gpu-kernel-correctness`)

1. **`TimeOnState` duplicado** — state kernels (E,IP,IS,H,ICU) já incrementam; `update_kernel`
   não deve incrementar (estava fazendo progressão 2× mais rápida entre estados)
2. **`AgeDays++` faltando** — original incrementa `AgeDays` em `Updatefunc()`; foi adicionado
3. **Substituição de morto com distribuição errada** — GPU usava 3 baldes (0-20,20-59,60-89);
   corrigido para rejection sampling com `ProbNaturalDeath` igual ao original
4. **Invariante `AgeDeathYears >= AgeYears`** — swap de idades não era feito após substituição
5. **`L=100`/`MAXSIM=5` hardcoded no `main()` do `covid.cu`** (commit `1c01ab2`) — o `main()`
   declarava `const int L = 100` e `const int MAXSIM = 5` locais ("small for testing"), fazendo
   *shadow* das globais de `define.h`. Resultado: a GPU **sempre** rodava L=100/MAXSIM=5,
   independentemente do `define.h` — toda validação anterior ficou presa em grid 100×100.
   Removidas as locais de `L`, `N` e `MAXSIM`; agora usa as globais (`gridSize` e `DAYS_TO_RUN`
   continuam locais).

---

## Benchmark São Paulo (L=3355, MAXSIM=10, 400 dias) — 2026-06-01

Primeira comparação serial × GPU no tamanho real de São Paulo (após corrigir o bug #5).
Resultados completos em `benchmarks/serial_runs.md` (Run #1 serial, #2/#3 GPU).

### ⏱️ Speedup (resultado sólido)

| Implementação | Tempo total (10 sims) | Por sim | Speedup |
|---------------|-----------------------|---------|---------|
| Serial (CPU, 1 thread) | ~97.7 min (5862 s) | ~9.8 min | 1× |
| GPU (RTX 4070 SUPER) | 67.5 s | ~6.75 s | **~87×** |

O speedup independe do Beta (tempo dominado pelo tamanho do grid).

### 📊 Resultados (dia 400) — divergem

| Métrica | Serial (β=0.0658) | GPU (β=0.0658) | GPU (β=0.0163) |
|---------|-------------------|----------------|----------------|
| Taxa de ataque (1−S) | 52.0% | **81.1%** | ~0% (extinta) |
| Recuperados | 44.5% | 81.0% | 0% |
| Mortes COVID | 6.15% | 12.68% | 0% |
| Pico TotalInfectious | 0.56% (dia 372) | 1.18% (dia 252) | 0 |

---

## Estado atual da validação — ✅ RESOLVIDO (ver `benchmarks/RESUMO.md`)

> **ATUALIZAÇÃO (jun/2026):** a divergência serial↔GPU foi **RESOLVIDA**. Eram **2 bugs no SERIAL**
> (a GPU estava correta): (1) `Update.h` — isolamento ligado por engano via `#if(BeginOfIsolation==ON)`
> avaliado pelo pré-processador; (2) `S.h` — `Sfunc` apagava infecções do infected-driven
> (`else Swap=S`). Com os fixes, **mesmo β → mesmas curvas nas 4 cidades** (RMSE de S(t) < 1,2 p.p.).
> Speedup serial×GPU: 1,8× (Rocinha) a **56× (SP)**. Benchmarks, profile, comparação entre GPUs e
> variabilidade **completos** → síntese em **`benchmarks/RESUMO.md`**.
> Branches: `r0` (serial + 2 fixes) · `fix/state-machine-timing` (validação + benchmarks).
>
> **O texto abaixo é HISTÓRICO (hipóteses da época, já superadas) — mantido por registro.**

### Histórico (pré-correção)

**Diagnóstico atualizado (com L grande, pós-fix #5):** com **mesmo Beta=0.0658**, a GPU produz uma
epidemia **mais intensa e mais rápida** que o serial (ataque 81% vs 52%, pico ~120 dias antes).
→ **A GPU transmite MAIS que o serial no mesmo Beta** — o mecanismo de spread da GPU está mais
"quente". (Obs.: a antiga observação "GPU lenta demais" era no regime L=100 com o bug #5 ativo;
descartada.)

**Sobre o Beta / calibração R0:** a calibração do branch `r0` (β=0.0163 → R0=3.5 no modo
paciente-zero) **NÃO transfere para o full-sim**: com β=0.0163 a epidemia full-sim **se extingue**
(R0_efetivo < 1). Provável causa: aquela medição de R0 foi **inflada** pelo bug conhecido (o
`S_kernel` não checava `PatientZeroID`, então infecções secundárias entravam na conta do paciente
zero). Conclusão: **0.0163 nunca deu R0=3.5 de verdade**, e o full-sim GPU precisa de calibração
própria (ou correção do spread antes).

**Fato-chave para o debug:** comparar sempre com **mesmo Beta e mesmo L**. No full-sim NÃO há
restrição de paciente-zero; o `d_Beta` é aplicado a TODA transmissão em `gpu_neighbors.cuh`:
- linha ~143: `1 - pow(1-d_Beta, infectiousContacts)` → contágio do suscetível (caminho S_kernel)
- linha ~119: `1 - pow(1-d_Beta, oldval+1)` → contágio via infectado (spreadInfection)

**Hipótese principal (a investigar):** dupla contagem / falha de deduplicação. Um suscetível pode
ser infectado no mesmo passo pelo **S_kernel** (ele procura infectados) **e** pelo
**spreadInfection** (um infectado mira nele). Se `Checked`/`Exponent` não estão evitando isso na
GPU como no serial, a GPU super-transmite → explica 81% vs 52%.

**Próxima ação combinada:** diff de mecanismo lado a lado:
`Neighbors.h`+`Neighborsinfected.h` (serial) × `checkAllContacts`+`spreadInfection` (GPU),
focando em `Checked`/`Exponent` e na ordem dos kernels no dia.

---

## Como compilar e rodar

```powershell
# Adicionar cl.exe ao PATH primeiro:
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64;$env:PATH"

cd CudaRuntime1/CudaRuntime1

# GPU (simulação completa)
nvcc covid.cu -o covid_sim.exe -arch=sm_89 --diag-suppress 20091
.\covid_sim.exe

# Serial
nvcc kernel.cu -o serial_sim.exe -arch=sm_89 --diag-suppress 20091 -Xcompiler "/wd4716"
.\serial_sim.exe

# Testes unitários
nvcc test_runner.cu -o test_runner.exe -arch=sm_89
.\test_runner.exe   # deve mostrar: ALL TESTS PASSED (15/15)
```

Parâmetros (apenas em `define.h` — após o fix #5, `covid.cu` lê `L`/`N`/`MAXSIM` das globais):
- `L = 100`, `MAXSIM = 5` para comparação/depuração rápida serial/GPU (roda em segundos)
- `L = 3355`, `MAXSIM = 10` para simulação completa de São Paulo (serial ~98 min, GPU ~68 s)
- Cidade: serial em `kernel.cu` (`cities(SP)`); GPU em `covid.cu` (`int city = SP`)

GPU: NVIDIA RTX 4070 SUPER (sm_89, compute 8.9)

---

## Saídas geradas

- `epidemicsprevalence.dat` — média de prevalência (proporção por estado)
- `epidemicsincidence.dat` — média de incidência (novos casos por dia)
- `Infectiousprevalence/incidence.dat` — detalhe ISLight/Moderate/Severe
- `prevalence_N.dat` / `incidence_N.dat` — dados brutos por simulação N
- `parameters.out` — parâmetros usados (serial only)

Colunas: `dias S E IP IA TotalInfectious H ICU Recovered DeadCovid`

- `benchmarks/serial_runs.md` — log de cada rodada (parâmetros, tempo/benchmark, sumário e
  comparação serial×GPU). **Atualizar a cada nova execução** (prática combinada com o autor).

---

## Próximos passos

1. ✅ **Divergência do spread diagnosticada e corrigida** — eram 2 bugs no serial (isolamento via
   `#if`; Sfunc apagando infecções). Serial≡GPU no mesmo β. Ver `benchmarks/equivalencia_state_machine.md`.
2. ✅ **Validação das 4 cidades** — serial e GPU batem no mesmo β (RMSE < 1,2 p.p.).
3. ✅ **Benchmark de performance** — 4 cidades (1,8×–56×), eficiência, escala×L, block size, profile.
4. ✅ **Comparação entre GPUs** (4070S × 1050 Ti) e **análise de variabilidade** (50 sims/cidade).
5. ⬜ **Abrir PR** no GitHub: `ViniciusAnjos/CUDAcovid-simulation` (consolidar `fix/state-machine-timing`).

> Síntese completa de tudo: **`benchmarks/RESUMO.md`**.
