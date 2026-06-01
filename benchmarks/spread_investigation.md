# Investigação da divergência do spread (serial × GPU)

Branch: `fix/gpu-spread-divergence` (a partir de `fix/gpu-kernel-correctness`).
Objetivo: descobrir por que, com **mesmo Beta e mesmo L**, a GPU transmite mais que o serial
(ataque 81% vs 52% em SP L=3355) — isolar e corrigir o mecanismo de spread.

## Baseline de depuração

`define.h` (serial e GPU iguais, exceto nada relevante):
- `L = 100`, `MAXSIM = 5` (iteração em segundos)
- `Beta = 0.0658` (mesmo dos dois — comparação justa do mecanismo)
- `DAYS = 400`

---

## Verificação de constantes (feita ANTES de investigar) — 2026-06-01

Conferido que tudo que alimenta a simulação é igual entre serial (`define.h`/`cities.h`/`death.h`/
`probsrecovery.h`/`agestructure.h`) e GPU (`define.h`/`gpu_define.cuh::buildArrays`/`setupCityParameters`).

### ✅ Idênticos
- `define.h` inteiro (Beta, probabilidades de transição, durações dos estados, condições iniciais
  IPini=5, taxas de ocupação 50%) — só diferem L/MAXSIM, controlados.
- Parâmetros de cidade SP: BEDSPOP=0.00247452, ICUPOP=0.00043782, Density=HIGH.
- `ProbNaturalDeath[0..120]` — valor a valor.
- Estrutura etária: `ProbBirthAge`, `SumProbBirthAge`, `AgeMin/Max`.
- `ProbRecoveryModerate` e `ProbRecoverySevere` (idades 0–89).

### ⚠️ DIFERENÇAS encontradas (PENDÊNCIA — resolver depois)

Recuperação de idosos **90+** em Hospital/UTI:

| Array (idade 90+) | Serial | GPU |
|---|---|---|
| `ProbRecoveryH` | `ProbRecoverySevere_Greater90` = **0.00167** | `ProbRecoveryH_Greater90` = **0.477** |
| `ProbRecoveryICU` | `ProbRecoverySevere_Greater90` = **0.00167** | `ProbRecoveryICU_Greater90` = **0.08333** |

- Causa: o serial (`probsrecovery.h` linhas 68 e 86) usa `ProbRecoverySevere_Greater90` para H e ICU
  em 90+ — aparenta ser **copy-paste bug** no serial. A GPU usa a constante "certa".
- Faixa de índice: serial preenche `[90,119]`; GPU preenche `[90,120]` (GPU inicializa o índice 120,
  serial deixa sem inicializar) em Moderate/Severe/H/ICU.
- **Impacto no spread ≈ zero**: afeta só ~0.38% da população (90+), e em H/UTI, que não transmitem.
- **Decisão pendente:** para validação estrita serial↔GPU, ou a GPU replica o serial como está
  (bug incluso), ou corrige os dois. Não bloqueia a investigação do spread.

---

## Mudança: contatos aleatórios de SP → documentação (2–19)

Antes: `MinRandomContacts=1.5`, `MaxRandomContacts=2.5` (≈ 1–2 contatos) — **não** batia com a doc.
A doc (CLAUDE.md) define SP = **2–19**.

Aplicado em **ambos** (mantendo serial = GPU):
- Serial: `cities.h` (árvore r0)
- GPU: `gpu_define.cuh::setupCityParameters` (worktree)

`MaxRandomContacts: 2.5 → 18.5`, `MinRandomContacts` mantido em `1.5`.

Convenção: segue a Rocinha (doc "2–120" → código `1.5/119.5`), i.e. `low−0.5 / high−0.5`.
→ SP "2–19" → `1.5 / 18.5`.

**Pendências/notas:**
- **Beta:** `0.0658` foi calibrado para SP com os contatos antigos (~2). Com 2–19 (média ~10),
  a transmissão sobe → R0 efetivo do serial **não é mais 3.5**. Recalibrar o Beta depois.
  (Não atrapalha a investigação do spread, que compara serial×GPU nos mesmos parâmetros.)
- **Truncamento:** o código faz `int RandomContacts = rn*(Max−Min)+Min` (trunca). Com `1.5/18.5`
  o inteiro real fica **1..18** (igual à Rocinha `1.5/119.5` → 1..119). Se quisermos o literal
  **2..19**, usar `Min=2.0, Max=20.0`. — decisão pendente.

---

## Hipótese principal a investigar

Com mesmo Beta, GPU transmite mais que serial. Suspeita nº 1: **dupla contagem / falha de
deduplicação**. Um suscetível pode ser infectado no mesmo passo:
1. pelo `S_kernel` (`checkAllContacts`) — ele procura infectados, e
2. pelo `spreadInfection_kernel` — um infectado mira nele.

Se `Checked`/`Exponent` não estão evitando isso na GPU como no serial (`Neighbors.h` +
`Neighborsinfected.h`), a GPU super-transmite → explica 81% vs 52%.

**Também verificar:** ordem de execução dos kernels no dia, e se a calibração R0 do branch `r0`
(0.0163) foi inflada por esse mesmo efeito no modo paciente-zero.

### Próximo passo
Diff de mecanismo lado a lado:
`Neighbors.h`+`Neighborsinfected.h` (serial) × `checkAllContacts`+`spreadInfection` (GPU).
