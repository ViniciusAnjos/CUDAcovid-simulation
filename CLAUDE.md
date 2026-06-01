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

## Calibração do Beta (branch `r0`)

**Objetivo:** encontrar β tal que R0 = 3.5 na implementação GPU.

**Definição de R0 aqui:** número médio de pessoas que o caso índice (paciente zero)
infecta **diretamente** em uma população 100% suscetível.

**Método correto:**
1. Inicializar com 1 IP (paciente zero), resto S
2. Rodar até paciente zero se recuperar (sair do estado IP)
3. Contar quantas pessoas passaram S→E por causa direta do paciente zero
4. Esse número = R0 desta rodada
5. Repetir muitas vezes (MAXSIM alto), tirar a média
6. Ajustar Beta até média ≈ 3.5

**Estado atual da implementação (`PATIENT_ZERO_ONLY_MODE`):**
- `gpu_define.cuh`: `#define PATIENT_ZERO_ONLY_MODE 1` — ativo
- `gpu_person.cuh`: campo `PatientZeroID` adicionado à struct GPUPerson
- `gpu_begin.cuh`: força exatamente 1 IP com `PatientZeroID = 1`
- `gpu_neighbors.cuh`: `spreadInfection_kernel` só executa para `PatientZeroID == 1`

**Bug no modo atual:**
- `S_kernel` (`checkAllContacts`) NÃO verifica `PatientZeroID` — susceptíveis
  ainda encontram e são infectados por casos secundários via contatos aleatórios
- Resultado: epidemia se propaga além do caso índice, inflando o R0 medido
- **Não existe nenhum contador que meça o R0** — o código salva apenas os
  arquivos `.dat` normais de prevalência, sem reportar o R0 calculado

**O que precisa ser implementado:**
- Contador `d_R0_count` incrementado só quando paciente zero causa S→E
- Leitura do contador após paciente zero se recuperar
- Loop de calibração: rodar N vezes, calcular média, ajustar Beta

---

## Bugs já corrigidos (branch `fix/gpu-kernel-correctness`)

1. **`TimeOnState` duplicado** — state kernels (E,IP,IS,H,ICU) já incrementam; `update_kernel`
   não deve incrementar (estava fazendo progressão 2× mais rápida entre estados)
2. **`AgeDays++` faltando** — original incrementa `AgeDays` em `Updatefunc()`; foi adicionado
3. **Substituição de morto com distribuição errada** — GPU usava 3 baldes (0-20,20-59,60-89);
   corrigido para rejection sampling com `ProbNaturalDeath` igual ao original
4. **Invariante `AgeDeathYears >= AgeYears`** — swap de idades não era feito após substituição

---

## Calibração R0 — progresso (branch `r0`)

### ✅ CALIBRAÇÃO CONCLUÍDA: **Beta = 0.0163 → R0 = 3.50** (alvo atingido)

`define.h` está com `const double Beta = 0.0163`.

### Resultados confirmados (código atual, pós-fix early-stopping, 1000 sims)
| Beta   | R0 médio | Confiável |
|--------|----------|-----------|
| 0.0138 | 2.98     | ✓         |
| 0.0144 | 3.10     | ✓         |
| 0.0147 | 3.17     | ✓         |
| 0.0150 | 3.24     | ✓         |
| 0.0162 | 3.48     | ✓         |
| **0.0163** | **3.50** | ✓ (alvo) |

> ⚠️ Os valores antigos (0.0150→4.42, 0.0200→4.66, etc.) foram medidos **antes** do
> fix de early-stopping/reset e estavam inflados. A re-medição de 0.0150 com o código
> atual deu 3.24, não 4.42 — descartar a tabela antiga. A relação β×R0 é ~linear nesta
> faixa, slope ≈ 217/unidade de β.

### Implementação da calibração (branch `r0`)
- `PATIENT_ZERO_ONLY_MODE` ativo em `gpu_define.cuh`
- `d_R0_count` — contador de filhos diretos do paciente zero
- `d_patientZeroActive` — flag para early stopping
- **Early stopping corrigido:** para quando paciente zero sai de IP/IA/ISLight
  (não espera recuperação total — evita rodar H/ICU desnecessariamente)
- `resetForR0_kernel` + `placePatientZero_kernel` — reset barato por simulação
  (init de idades feito apenas 1x antes do loop)

### Problema operacional encontrado
O PowerShell tool sempre roda em background, acumulando múltiplos processos
`covid_sim` simultâneos com race condition em `define.h` e `covid_sim.exe`.
**Solução para próxima sessão:** testar um Beta por vez, esperar notificação
de conclusão antes de disparar o próximo.

### Workflow correto para calibração
```
# 1. Compilar uma vez com o Beta desejado
(Get-Content define.h) -replace 'Beta = [^;]+', "Beta = 0.0125" | Set-Content define.h
nvcc covid.cu -o covid_sim.exe -arch=sm_89 --diag-suppress 20091

# 2. Rodar e esperar conclusão (notificação automática)
.\covid_sim.exe

# 3. Ler resultado e decidir próximo Beta
# R0 < 3.5 → aumentar Beta; R0 > 3.5 → diminuir Beta
```

---

## Estado atual da validação (simulação completa)

**Problema em aberto:** Com L=100, MAXSIM=5, os resultados serial e GPU divergem:
- Serial: epidemia explode rapidamente (S cai para ~0.06 no dia 25)
- GPU: propagação muito lenta (S ainda ~0.98 no dia 25)

**Hipóteses a investigar:**
1. `spreadInfection_kernel` pode estar com conflito de `Checked`/`Exponent` com `S_kernel`
2. RNG per-thread com L pequeno pode gerar viés
3. Ordem de execução dos kernels pode fazer `Checked=1` do `S_kernel` bloquear
   o `spreadInfection_kernel` no mesmo timestep

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

Parâmetros de teste rápido (em `define.h` e `covid.cu` main):
- `L = 100`, `MAXSIM = 5` para comparação rápida serial/GPU
- `L = 3200`, `MAXSIM = 10` para simulação completa

GPU: NVIDIA RTX 4070 SUPER (sm_89, compute 8.9)

---

## Saídas geradas

- `epidemicsprevalence.dat` — média de prevalência (proporção por estado)
- `epidemicsincidence.dat` — média de incidência (novos casos por dia)
- `Infectiousprevalence/incidence.dat` — detalhe ISLight/Moderate/Severe
- `prevalence_N.dat` / `incidence_N.dat` — dados brutos por simulação N
- `parameters.out` — parâmetros usados (serial only)

Colunas: `dias S E IP IA TotalInfectious H ICU Recovered DeadCovid`

---

## Próximos passos

1. **Diagnosticar divergência serial/GPU** — investigar `spreadInfection_kernel` vs `S_kernel`
   e o uso de `Checked`/`Exponent` para evitar dupla infecção
2. **Calibrar Beta na GPU** — verificar se β=0.0658 produz R0≈3.5 na implementação GPU
3. **Benchmark de performance** — medir speedup GPU vs serial para L=3200
4. **Testar todas as 4 cidades** — SP, Rocinha, Brasília, Manaus
5. **Abrir PR** no GitHub: `ViniciusAnjos/CUDAcovid-simulation`
