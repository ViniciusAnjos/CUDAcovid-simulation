# Investigação da diferença de duração infecciosa (serial × GPU)

Branch: `fix/gpu-spread-divergence` (worktree). Motivo: a validação das epidemias completas
(`r0_calibration_gpu.md`) mostrou que **calibrar R0 não faz serial e GPU baterem** — a GPU cresce
muito mais rápido. Causa provável: **tempo de geração menor** porque a GPU tem ~12% menos
agente-dias infecciosos (`spread_investigation.md` Exp 9). Objetivo: achar e corrigir essa diferença
de duração para que R0 **e** tempo de geração coincidam.

## Mapa de código (resets e transições de TimeOnState)

`TimeOnState = 0` (reset) — **idêntico** serial × GPU:
- init população (begin.h:86 / gpu_begin.cuh:59)
- infecção S→E susceptível-driven (S.h:21 / S_kernel.cuh:26)
- infecção S→E infected-driven (Neighborsinfected.h:47 / gpu_neighbors.cuh:127)
- reposição de morto → novo S (Update.h:55 / Update_kernel.cuh:98)

**Nenhum** dos dois reseta TimeOnState nas transições E→IP→IA/IS. Lógica de transição idêntica:
`TimeOnState++` no início do kernel/func, depois `if (TimeOnState >= StateTime)`. `StateTime` (int)
re-sorteado a cada transição. `E.h` == `E_kernel.cuh`, `IP.h` == `IP_kernel.cuh`, etc. (verificado).

> Consequência do design (em AMBOS): como TimeOnState só zera na infecção e cresce sem reset, após o
> 1º estado (E) os seguintes (IP→IA→...) costumam durar ~1 dia (TimeOnState já é grande). Só o
> estado seedado (IP inicial com TimeOnState=0) dura sua duração cheia.

## Hipóteses a testar
- **H1 — duração real difere (medir):** Beta=0, semear 2000 IP, comparar integral de IP/IA/ISLight.
- **H2 — RNG per-thread enviesa StateTime:** se o LCG per-thread dá rn médio < 0.5, StateTime=(int)(rn·14.5)
  vem menor → durações menores. (Seedados usam rngStates[0]; transições usam rngStates[idx].)
- **H3 — truncamento/`(int)` difere** em algum sorteio de StateTime.
- **H4 — morte natural/idade difere** encurtando o período.
- **H5 — contagem/janela de prevalência difere** (off-by-one no dia de transição).

## Resultados
_(preenchido conforme as medições)_

### Exp A — duração por estado (Beta=0, IPini=2000, L=200, MAXSIM=200)

| Estado (agente-dias) | Serial | GPU | razão |
|----------------------|--------|-----|-------|
| IP | 0.3296 | 0.2800 | 0.850 |
| IA | 0.0329 | 0.0329 | 1.001 |
| ISLight | 0.0416 | 0.0418 | 1.004 |

**🎯 A DURAÇÃO É IDÊNTICA. O "12%" era ARTEFATO DE REPORTE DO DIA 0.**
- IA e ISLight idênticos. Só o IP difere, e exatamente pelo valor do dia 0:
  serial IP(dia 0) = 2000/40000 = **0.05**; `0.3296 − 0.05 = 0.2796 ≈ 0.2800` (GPU).
- 🐛 **Bug de reporte (dia 0):** `distributeInitialInfections_kernel` atualiza o array
  `d_stateCounts` (atomicAdd em stateCounts[d_IP]), mas `getCountersFromDevice` lê os globais
  `d_IP_Total` etc., que `initSimulationCounters` zerou e só são preenchidos pelo `update_kernel`
  a partir do dia 1. Logo a GPU **não reporta as infecções semeadas no dia 0**.
- **Impacto:** só no REPORTE (.dat), não na dinâmica (a população tem os IP e eles espalham/transicionam
  normalmente). Para a epidemia real (IPini=5) o sub-reporte do dia 0 é 5 agentes × 1 dia → desprezível.
- **Conclusão:** a hipótese de "tempo de geração diferente" (duração infecciosa) está **REFUTADA**.
  Não há diferença de duração. A divergência das epidemias completas vem de OUTRA coisa.

> ⚠️ Reavaliar: a calibração de R0 deu GPU R0 menor que serial no mesmo Beta (3.18 vs 3.52), o que
> motivou o Beta maior da GPU. Se a duração é igual, esse R0 menor não vem da duração — investigar
> se é artefato do medidor r0_gpu OU diferença real de transmissão/dia. Próximo: comparar epidemia
> serial × GPU no MESMO Beta (sem o confundidor da calibração).

### Exp B — epidemia no MESMO Beta (β=0.0213, SP, L=300, IPini=5, MAXSIM=5)

| dia | S serial | S GPU | Inf serial | Inf GPU |
|----:|---------:|------:|-----------:|--------:|
| 30 | 0.9993 | 0.9986 | 0.0000 | 0.0000 |
| 80 | 0.9969 | 0.9766 | 0.0001 | 0.0006 |
| 200 | 0.9696 | 0.4658 | 0.0004 | 0.0053 |
| 400 | 0.7903 | 0.3446 | 0.0013 | 0.0001 |
| **Ataque** | **21%** | **66%** | | |

**No MESMO Beta, GPU ≈ 3× serial.** A divergência é mecanismo real (não a calibração), e compõe —
bate com o "~3× em β baixo" do `spread_investigation.md` (Exp 8). A calibração de R0 PIOROU porque
deu Beta maior à GPU.

### 🔴 Contradição reveladora (próxima pista)
- Medidor de R0 (1 paciente-zero): GPU **3.18** < serial **3.52** no mesmo Beta.
- Epidemia (alta prevalência): GPU **66%** ≫ serial **21%** no mesmo Beta.
- R0 menor + epidemia maior é **impossível** se R0 governasse → a diferença de transmissão GPU×serial
  **depende da prevalência**: a baixa prevalência (1 caso) a GPU transmite MENOS; a alta prevalência
  transmite MAIS. Isso aponta para o **canal suscetível-driven de contatos aleatórios** se comportar
  diferente conforme a densidade de infectados (já localizado lá na investigação do spread).
- **Hipótese nova (H6):** a interação entre os DOIS mecanismos (S_kernel suscetível-driven roda
  TODOS antes; depois IP/IS infected-driven) com a deduplicação `Checked`/`Exponent` se comporta
  diferente do serial (interleaved por célula) **em função da prevalência** — efeito invisível com 1
  caso, dominante na epidemia. Próximo: medir transmissão/passo a prevalência controlada (ex.: semear
  X% infecciosos fixos, Beta baixo, contar New_E/dia) para várias prevalências X.
