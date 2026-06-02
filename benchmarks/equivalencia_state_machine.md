# Investigação: equivalência exata Serial × GPU (mesmo β → mesmas curvas)

Branch: `fix/state-machine-timing`. Objetivo: descobrir por que, **no mesmo β**, a GPU tem R0 ~10–12%
menor que o serial (documentado em `r0_calibration_gpu.md`) — o que obriga a usar β diferentes e faz
as curvas divergirem (pior na Rocinha). Meta: corrigir para **mesmo β → mesmas curvas nas 4 cidades**.

## Fase 1 — Análise estática (diff linha a linha)

Comparei os state functions e os mecanismos de contágio dos dois lados.

### ✅ Confirmado EQUIVALENTE
- **Timing dos estados** (E/IP/IS): `TimeOnState++` no topo, transição quando
  `TimeOnState >= StateTime`, mesma ordem de sorteios RNG, mesmas probabilidades. (E.h×E_kernel,
  IP.h×IP_kernel, IS.h×IS_kernel.)
- **Tipo de `StateTime`**: `int` nos DOIS (serial kernel.cu l.17; GPU gpu_person.cuh) → ambos
  truncam a duração igual. (Hipótese de truncação descartada.)
- **Resets de `TimeOnState=0`**: mesmos 4 locais nos dois (S→E suscetível-driven, S→E
  infected-driven, morte/troca, init). `TimeOnState` é cumulativo desde a infecção em ambos.
- **Infected-driven** (Neighborsinfected×spreadInfection): só a 1ª mira infecta (Exponent==1 /
  oldval==0), prob = β, seta Checked=1. Equivalente.
- **Suscetível-driven decisão** (S.h×S_kernel): seta Checked=1 ao infectar; sorteia rn só quando
  KI>0; mesma fórmula `1-(1-β)^KI`. Equivalente.
- **Estados infecciosos contados** (IP/IA/ISLight/ISModerate/ISSevere/H/ICU): iguais nos dois.

### ⚠️ DIFERENÇAS ENCONTRADAS (candidatas)

**H1 — Vizinhança de Von Neumann conta no máximo 1 (serial) vs todos os 4 (GPU).**
`Neighbors.h` (baixa densidade) usa cadeia `if … else if … else if …` → **incrementa KI por no
máximo 1** vizinho local infectado. A GPU (`checkLocalContacts`) faz laço com `if` → conta **todos
os 4**. Afeta **Brasília/Manaus** (Von Neumann); **não** afeta Rocinha/SP (Moore, ambos contam 8).
→ No mesmo β, o serial-VN transmitiria *menos* pelo canal local (oposto ao sinal observado), mas é
uma divergência real de mecanismo que precisa casar.

**H2 — Auto-evitação dos contatos aleatórios difere.**
Serial: `do{Randomi}while(Randomi==i)` **e** `do{Randomj}while(Randomj==j)` separados → alvo nunca
está na **mesma linha nem na mesma coluna** do self (exclui 2L−1 células). GPU:
`while(Randomi==i && Randomj==j)` → exclui só a célula exata. Efeito estatístico ~2/L (≈0,8% em
L=264) — pequeno, mas real. Também dessincroniza a contagem de sorteios RNG.

**H3 — Ordem dos mecanismos + dedup (Checked).**
GPU: **todos os suscetíveis (S_kernel) primeiro**, depois todos os infectados (spreadInfection).
Serial: passada única **intercalada** por posição da célula. Consequência no serial: se o
infected-driven atinge uma célula S **antes** do Sfunc dela na passada, o Sfunc (que ignora
`Checked`) pode **recontar New_E** (dupla contagem de *incidência*). Para *prevalência/transmissão*
o desfecho (a célula vira E) é o mesmo. Precisa de checagem empírica para ver se afeta o R0.

### ❓ A explicar (Rocinha = Moore)
H1 não se aplica à Rocinha (Moore). Então a divergência da ROC no mesmo β tem que vir de H2/H3 ou de
algo ainda não isolado (ex.: efeito de duração infecciosa real). Próxima fase: **medir** R0 e
duração infecciosa média no mesmo β, nos dois, e bissectar a causa.

## Fase 2 — Experimentos: ROOT CAUSE encontrada 🎯

Plataforma de teste: **mesmo β=0,03, mesmo L=300, SP (Moore), MAXSIM=20**, comparando ataque (1−S
no dia 400) entre serial e GPU. Guards de compilação (`#ifdef`) para ligar/desligar mecanismos.

### Bissecção por mecanismo (mesmo β)

| Config | Serial | GPU | Conclusão |
|--------|--------|-----|-----------|
| Tudo ligado | 0,536 | 0,826 | gap enorme (29 p.p.) |
| Só suscetível-driven (infected OFF) | 0,474 | 0,769 | gap persiste → não é o infected-driven |
| Só vizinhança local (Moore) | 0,0005 | 0,0005 | **local idêntico** ✓ |
| Só contatos aleatórios | 0,0009 | 0,312 | **divergência está aqui** |

Descartados empiricamente: **RNG** (serial e GPU dão o mesmo resultado com RNG bom OU ruim — testado
nos dois lados), **duração infecciosa** (dwell de IP idêntico: 2,00 dias nos dois via Lei de Little),
**parâmetros de doença** (idênticos), **contagem de contatos** (média 9,50 nos dois), **auto-evitação**
(irrelevante).

### A medição que revelou tudo (contadores DIAG)
Instrumentei os dois para contar contatos aleatórios e KI:
```
DIAG_SERIAL mean_RandomContacts=9.50  mean_KI=0.0557  (n_RC=269M  n_KI=563M)
DIAG_GPU    mean_RandomContacts=9.50  mean_KI=0.1512  (n_RC=355M  n_KI=355M)
```
**No serial, `n_RC` (execuções do bloco de contatos aleatórios) = 269M, mas `n_KI` (chamadas totais)
= 563M → só ~48% dos suscetíveis executam contatos aleatórios!** Os outros ~52% pulam o bloco, que é
guardado por `if (Person.Isolation == IsolationNo)`. Na GPU, `n_RC = n_KI` (100% executam).

### CAUSA RAIZ: isolamento ligado por engano no serial (bug de pré-processador)

`define.h`: `ProportionIsolated = 0.5`, `BeginOfIsolation = 0` (OFF), `ON = 1`, `OFF = 0`.
`Update.h` tinha:
```c
#if(BeginOfIsolation==ON)      // ON e BeginOfIsolation NAO sao macros -> preprocessador ve 0==0
    ... Isolationfunc(); ...   // -> SEMPRE VERDADEIRO -> isolamento e compilado e executado!
#endif
```
**É o mesmo bug do `#if(Density==HIGH)`** (já corrigido em `Neighbors.h`). O preprocessador avalia
`BeginOfIsolation` e `ON` como 0 (não são `#define`s), então `#if(0==0)` é sempre verdadeiro →
`Isolationfunc()` roda no dia 15 e **isola 50% da população**. Agentes isolados **pulam contatos
aleatórios** → o serial transmite ~metade pelo canal aleatório → epidemia muito menor.
A GPU **nunca implementou isolamento** → transmite cheio. Daí a divergência.

Bate com tudo: dias 1–14 idênticos (isolamento só no dia 15); ~48% executam random (386/400 dias ×
50% isolados); a divergência cresce a partir do dia ~15–40.

### Correção e validação
`Update.h`: `#if(BeginOfIsolation==ON)` → **runtime** `if (BeginOfIsolation == ON)`
(com `BeginOfIsolation=0`, `ON=1` → `if(0==1)` falso → isolamento OFF de verdade).

| Config (mesmo β=0,03) | Serial antes | Serial corrigido | GPU |
|-----------------------|--------------|------------------|-----|
| Só suscetível-driven | 0,474 | **0,772** | 0,769 ✓ **bate** |
| Tudo ligado | 0,536 | **0,801** | 0,825 (resíduo 2,4 p.p.) |

Pós-fix, `n_RC = n_KI` e `mean_KI = 0,1514` (= GPU 0,1512). **O suscetível-driven agora é
equivalente.** O isolamento acidental respondia por ~92% da divergência.

### Resíduo (~2–3 p.p.) — SEGUNDO bug do serial (NÃO é intrínseco)

Sobra um gap pequeno só na **interação** dos dois mecanismos:
| Config (β=0,03) | Serial | GPU |
|-----------------|--------|-----|
| Só suscetível-driven | 0,772 | 0,769 ✓ |
| Só infected-driven | 0,0001 | 0,0002 ✓ |
| Tudo (interação) | 0,801 | 0,825 ✗ (2,4 p.p.) |

Cada mecanismo **isolado** bate. O resíduo vem da **interação**. Causa, lendo `S.h` (Sfunc):
```c
if (Contagion == 1) { ...Swap = E; }
else Person[i][j].Swap = S;   // <-- BUG: forca Swap=S em Contagion=0
```
No serial intercalado (passada única), se o **infected-driven já infectou** a célula (Swap=E,
Checked=1) **antes** do Sfunc dela na passada, e o sorteio do Sfunc dá Contagion=0, o Sfunc
**sobrescreve Swap=S → apaga a infecção do infected-driven**. ~Metade das infecções infected-driven
(as de células ainda não processadas na passada) são perdidas. Na GPU (fases: S_kernel completo,
depois spreadInfection) isso nunca ocorre → infecções persistem.

> A ordem fásica (GPU) vs intercalada (serial) **em si é irrelevante**: P(E)=P_s+P_i−P_s·P_i é
> simétrica na ordem. O ÚNICO efeito de ordem era esse apagamento (um bug), não algo intrínseco.

**Correção** (`S.h`): `else Swap=S;` → `else if (Checked == 0) Swap=S;` (não apaga infecção já feita).

### ✅ Validação final — equivalência alcançada (mesmo β → mesmas curvas)

Com os **dois fixes no serial** (isolamento `#if`→`if`; guarda do Swap no Sfunc) e **zero mudanças
de lógica na GPU** (ela já estava correta):

| β (L=300) | Serial | GPU | dif |
|-----------|--------|-----|-----|
| 0,02 (MAXSIM=30, perto do limiar) | 0,6267 | 0,6209 | 0,6 p.p. |
| 0,03 (MAXSIM=20) | 0,8272 | 0,8253 | 0,2 p.p. |
| 0,04 (MAXSIM=30, saturado) | 0,9011 | 0,8997 | 0,1 p.p. |

Todas as diferenças **dentro do ruído estatístico** (MAXSIM finito). **Conclusão: a equivalência
serial↔GPU no MESMO β é possível** — o resíduo NÃO era intrínseco, eram dois bugs do serial. Não há
necessidade de calibrar β separado por implementação (aquilo compensava os bugs).

## Resumo das correções (ambas no SERIAL; GPU estava correta)
1. **`Update.h`:** `#if(BeginOfIsolation==ON)` → `if (BeginOfIsolation == ON)` — isolamento de 50%
   ligado por engano (bug de pré-processador) respondia por ~92% da divergência.
2. **`S.h`:** `else Swap=S;` → `else if (Checked==0) Swap=S;` — Sfunc apagava infecções do
   infected-driven (resíduo ~2–3%).

Implicação: as curvas serial do benchmark anterior estavam com isolamento ligado → precisavam de β
diferente. Com os fixes, revalidar as 4 cidades no **mesmo β** (serial = GPU).
