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

---

## Diff de mecanismo — leitura do código (2026-06-01)

### O que está IGUAL (verificado linha a linha)
- **Susceptível-driven** (`Neighbors.h` × `checkAllContacts`/`S_kernel`): conta 8 vizinhos Moore +
  contatos aleatórios infecciosos (KI), `P = 1-(1-β)^KI`, rola 1×. Estados infecciosos contados
  iguais (IP, IA, ISLight, ISModerate, ISSevere, H, ICU). ✓
- **Infected-driven** (`Neighborsinfected.h` × `spreadInfection_kernel`): para cada suscetível alvo,
  `Exponent++` (GPU usa `atomicAdd`), gate `Checked==0 && Exponent==1` (GPU: `Checked==0 && oldval==0`,
  equivalente), prob = β, escreve `Swap=E`, `Checked=1`. ✓
- **Quem espalha (infected-driven):** só IP, IA, ISLight, no ramo "ainda infeccioso" (else do
  TimeOnState). ISModerate/ISSevere/H/ICU **não** espalham. GPU idêntico (IP_kernel, IS_kernel). ✓
- **Double-buffer:** todos os kernels leem `Health`, escrevem `Swap`; `Health=Swap` só no update. ✓

### Hipóteses (a testar empiricamente)
- **H1 — chamadores do spread diferentes:** ❌ DESCARTADA na leitura (IP/IA/ISLight só, igual serial).
- **H2 — ordem dos kernels:** serial processa células interleaved em ordem; GPU roda `S_kernel`
  (TODOS suscetíveis) e só depois IP/IS (infected-driven). Pode mudar a probabilidade efetiva de
  infecção por suscetível (acoplamento via `Checked`).
- **H3 — RNG per-thread (LCG) enviesado:** poderia inflar nº de contatos ou alvos. Ver `gpu_aleat.cuh`.
- **H4 — reset de `Exponent`/`Checked` no update_kernel:** se não reseta, dedup quebra (mas isso
  esfriaria, não esquentaria — provavelmente não é a causa do GPU mais quente).
- **H5 — indexação de vizinhos/contorno conta nº errado de vizinhos** (KI inflado na GPU).
- **H6 — população inicial / período infeccioso difere** (mais infecciosos ou por mais dias).

### Experimentos planejados
1. **Reproduzir a divergência em L=100** (rápido): rodar serial e GPU com mesmos parâmetros,
   comparar curva S(t) e taxa de ataque.
2. **Ablação:** desligar o infected-driven (spreadInfection) nos dois e comparar — isola se a
   divergência está no susceptível-driven ou no infected-driven.
3. Conforme resultado, investigar RNG (H3), vizinhança (H5) ou ordem (H2).

### Resultados dos experimentos

#### Exp 1 — Baseline L=100 (β=0.0658, contatos 1.5/18.5, MAXSIM=5)
Tempo: serial 11.4 s, GPU 14 s.

| | S final | Ataque |
|---|---|---|
| Serial | 0.1308 | 86.9% |
| GPU | 0.0648 | **93.5%** |

Curva inicial S(t) e E(t) (prevalência):

| dia | S serial | S gpu | E serial | E gpu |
|----:|---------:|------:|---------:|------:|
| 1 | 0.99874 | 0.99884 | 0.00076 | 0.00070 |
| 5 | 0.99536 | 0.99580 | 0.00374 | 0.00330 |
| 8 | 0.99268 | 0.99154 | 0.00582 | 0.00680 |
| 10 | 0.98972 | 0.98736 | 0.00812 | 0.01018 |
| 15 | 0.97844 | 0.97110 | 0.01656 | 0.02338 |
| 25 | 0.93218 | 0.87004 | 0.04774 | 0.09930 |
| 30 | 0.89272 | 0.75210 | 0.07230 | 0.18476 |
| 40 | 0.75214 | 0.37396 | 0.15500 | 0.41448 |

**Leitura:** dias 1–5 quase idênticos; a GPU começa levemente *atrás* e **ultrapassa ~dia 8**,
com a diferença **compondo exponencialmente**. Padrão típico de **R0 efetivo da GPU um pouco
maior** — diferença sutil e sistemática por passo, não erro grosseiro. Descarta erro gritante de
contagem (senão divergiria já no dia 1).

**Diferenças estruturais candidatas encontradas na leitura:**
- **Auto-evitação dos contatos aleatórios:** serial re-sorteia até `Randomi!=i` E `Randomj!=j`
  (exclui a linha i e a coluna j inteiras); GPU re-sorteia só se `Randomi==i && Randomj==j`
  (exclui apenas a célula exata). Efeito pequeno (exclui 2L−1 de L² células), mas é divergência.
- **Seed do RNG:** serial 1 stream global por simulação; GPU `states[idx]=seed*(idx+1)`
  (sementes correlacionadas; metade pares → LCG multiplicativo `*888121` degrada bits baixos).
- Algoritmo do LCG e contagem de vizinhos (8 Moore): idênticos. ✓

#### Exp 2 — Ablação (infected-driven OFF nos dois)

| | Ataque (infected ON) | Ataque (infected OFF) |
|---|---|---|
| Serial | 86.9% | 84.4% |
| GPU | 93.5% | 91.4% |
| **Gap GPU−serial** | **6.6 pp** | **7.0 pp** |

**Conclusão:** desligar o infected-driven NÃO fecha o gap → a divergência está no
**susceptível-driven** (`S_kernel`/`checkAllContacts`), não no infected-driven (que é equivalente,
como a leitura já indicava). Confirma H1 descartada e foca em H2/H3/H5 no canal suscetível-driven.

#### Exp 3 — Puro local (contatos aleatórios = 0 nos dois, infected-driven segue OFF)

| (random=0) | Ataque |
|---|---|
| Serial | 6.71% |
| GPU | 6.48% |

**🎯 RESULTADO CHAVE: serial e GPU BATEM com só vizinhos locais.** O gap some. Logo a divergência
está **inteiramente no caminho de CONTATOS ALEATÓRIOS** do suscetível-driven. O caminho local
(8 vizinhos) + roll está correto e equivalente.

Diferenças possíveis no caminho de contatos aleatórios:
- (a) **auto-evitação:** serial exclui linha i + coluna j; GPU exclui só a célula exata;
- (b) **RNG per-thread** no nº de contatos e/ou na seleção dos alvos.

#### Exp 4 — Contagem de contatos FIXA (Min=Max=10) nos dois
| (count fixo=10, infected OFF) | Ataque |
|---|---|
| Serial | 84.5% |
| GPU | 91.9% |

Gap **persiste** (~7.4 pp) → NÃO é o número de contatos. Sobra: seleção de alvo / RNG / auto-evitação.

#### Exp 5 — RNG com seeding por hash (em vez de `seed*(idx+1)`)
GPU: **91.9%** (inalterado). Gap persiste → NÃO é o seeding correlacionado do RNG.

#### Exp 6 — Auto-evitação da GPU igual ao serial (exclui linha i E coluna j)
GPU: **91.8%** (inalterado). Gap persiste → NÃO é a auto-evitação.

#### Exp 7 — Output-hash no `generateRandom` (decorrelaciona saída do LCG)
GPU: **91.86%** (inalterado). Gap persiste → NÃO é a qualidade do RNG.

**Pista decisiva:** GPU ficou em ~0.918 em EXP4/5/6/7 apesar de mudanças no seeding, output do RNG
e auto-evitação. Se a identidade dos alvos aleatórios importasse, mudar o RNG mudaria o resultado.
Não muda → o gap **não vem da amostragem aleatória**; é **determinístico**. Provável artefato de
**saturação/demografia** (a métrica "1−S_final" mistura suscetíveis regenerados por morte/substituição),
não diferença de R0 no spread.

#### Exp 8 — Beta baixo (β=0.01, não-saturante), count fixo=10, infected OFF
Curva (prevalência), L=100:

| dia | S serial | S gpu | E serial | E gpu |
|----:|---------:|------:|---------:|------:|
| 10 | 0.99874 | 0.99856 | 0.00058 | 0.00080 |
| 60 | 0.99744 | 0.99560 | 0.00032 | 0.00076 |
| 200 | 0.99562 | 0.98976 | 0.00002 | 0.00034 |
| 400 | 0.99562 | 0.98754 | 0 | 0 |

| | Ataque (dia 400) |
|---|---|
| Serial | 0.44% |
| GPU | **1.25% (~3×)** |

A epidemia serial **morre** (E→0 ~dia 200); a GPU **sustenta** transmissão por mais tempo.
→ **Diferença REAL de R0 efetivo** no canal de contatos aleatórios, **não** artefato de saturação.
E **determinística** (idêntica sob todas as mudanças de RNG/auto-evitação dos Exp 5–7).

---

## Conclusões parciais da investigação (2026-06-01)

**Localização (sólida):** a divergência serial×GPU está **inteiramente no canal de CONTATOS
ALEATÓRIOS do mecanismo suscetível-driven**. Confirmado por ablação:
- Infected-driven OFF → gap permanece (Exp 2). ❌ não é o infected-driven.
- Só vizinhos locais → serial e GPU **batem** (Exp 3). ✓ canal local correto.
- Contatos aleatórios ON → gap de ~3× (Exp 8). ← **aqui está o problema.**

**Descartado por experimento (gap NÃO muda):** número de contatos (Exp 4), seeding do RNG (Exp 5),
auto-evitação (Exp 6), qualidade/output do RNG (Exp 7). A diferença é **determinística**, não
estatística — mudar o RNG não muda o resultado.

**Hipótese líder restante (a testar):** o gap aparece só com **mistura global** (contatos aleatórios)
e some com spread puramente local. Isso é consistente com uma diferença no **tempo/pool infeccioso**
(quantos "agente-dias infecciosos" existem): no spread local a transmissão é limitada pelos 8
vizinhos (saturam rápido, dias extras não importam), mas no spread aleatório cada dia infeccioso a
mais = mais suscetíveis sorteiam aquele agente = mais transmissão. Uma diferença pequena na duração
infecciosa (IP/IA/ISLight) seria **amplificada ~linearmente** no canal aleatório e quase invisível
no local — batendo com Exp 3 vs Exp 8.
→ **Próximo passo:** medir os agente-dias infecciosos (duração média em IP/IA/ISLight) serial × GPU,
e revisar `TimeOnState`/`StateTime`/transições e a contagem do pool infeccioso no `update_kernel`.

**Alternativa a investigar:** diferença determinística na contagem de KI no caminho aleatório que
não dependa do RNG (ex.: alvo lido de um índice/estado sutilmente diferente).

> Notas: experimentos com L=100, MAXSIM=5, infected-driven desligado nos dois (ablação), contatos
> fixos em 10 (exceto Exp 1/3). Mudanças de código foram revertidas ao baseline após os testes;
> este documento é o registro.
