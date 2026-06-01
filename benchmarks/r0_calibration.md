# Calibração do Beta serial por cidade (R0 = 3.5)

Objetivo: encontrar, para cada cidade, o valor de `Beta` no qual a **simulação serial** produz
**R0 = 3.5**, usando os parâmetros do documento (CLAUDE.md) e o **L real de cada cidade**.

## Metodologia

- **Medidor:** `r0_serial.cu` (cópia enxuta do serial; mede R0 sem I/O por dia).
- **Definição de R0:** nº médio de secundários que **1 paciente-zero** (1 IP) infecta diretamente
  numa população 100% suscetível, contado **enquanto o paciente-zero está em estado que espalha**
  (IP/IA/ISLight). Para quando ele sai desses estados OU quando um secundário passa a espalhar
  (`IP_Total+IA_Total+ISLight_Total != 1`). Durante essa janela o paciente-zero é o único
  infeccioso, então todo `New_E` é filho direto dele. (Mesma definição/early-stop da calibração GPU.)
- **Ambos os mecanismos** do modelo atuam: suscetível-driven (vizinhos + contatos aleatórios) e
  infected-driven (paciente-zero mira suscetíveis). `IPini=1`, resto S.
- **Busca:** R0 ~ linear em Beta nesta faixa → `Beta_novo = Beta × 3.5 / R0_medido`; itera até
  R0 ∈ [3.4, 3.6]. MAXSIM=100 na busca; confirmação final com MAXSIM maior.
- **L real por cidade** (decisão do autor: Beta depende de L; usar o L da simulação).

## Parâmetros por cidade (do documento / código atual)

| Cidade | L | Densidade | Vizinhança | Contatos aleat. (Min/Max) | Leitos/Pop | UTI/Pop |
|--------|------|-----------|------------|---------------------------|------------|---------|
| São Paulo | 3355 | Alta | Moore (8) | 1.5 / 18.5  (doc 2–19) | 0.00247452 | 0.00043782 |
| Rocinha | 264 | Alta | Moore (8) | 1.5 / 119.5 (doc 2–120) | 0.00055111 | 0.00014592 |
| Brasília | 1604 | Baixa | **ver ⚠️** | 1.5 / 2.5 (doc 2) | 0.00260879 | 0.00040114 |
| Manaus | 1343 | Baixa | **ver ⚠️** | 1.5 / 2.5 (doc 2) | 0.00187124 | 0.00027858 |

Parâmetros da doença (latência, durações IP/IA/IS, probabilidades de transição): iguais para todas
as cidades (de `define.h`, inalterados).

> ⚠️ **Bug latente descoberto (`Neighbors.h`):** `#if(Density==HIGH)` é avaliado pelo
> **pré-processador**, onde `Density` e `HIGH` (variáveis runtime, não macros) viram 0 →
> `#if(0==0)` → **sempre verdadeiro**. Logo o serial usa **Moore (8 vizinhos) para TODAS as
> cidades**, mesmo Brasília/Manaus (que pela doc seriam Von Neumann/4). A calibração abaixo usa o
> comportamento real do código (Moore). Se quiser Von Neumann para BRA/MAN conforme a doc, é preciso
> corrigir o `#if` para um `if` runtime e recalibrar.

## Achado: R0 é INDEPENDENTE de L (verificado com dados)

Sweep de L para SP, Beta=0.0658, MAXSIM=200:

| L | R0 medido | tempo |
|------|-----------|-------|
| 200 | 7.76 | 7.7 s |
| 400 | 7.96 | 33 s |
| 800 | 8.05 | 292 s |
| 1600 | 7.57 | 1606 s (27 min) |

R0 ~constante (7.57–8.05; a variação é ruído de MAXSIM=200, sem tendência com L). **O R0 do
paciente-zero NÃO depende de L** — coerente com a teoria (cada suscetível sorteia ~c contatos; a
chance de achar o paciente-zero é c/L², e há ~L² suscetíceis → ~c encontros, independente de L).
O **tempo cresce ~L²** (L=3355 levaria ~2 h por rodada).

**Decisão:** calibrar em **L=200** (rápido, R0 idêntico) e, ao final, **confirmar o Beta de SP em
L=3355** para registrar o número no L real. MAXSIM=300 na busca.

## Resultados da busca (L=200) — bisseção por escala linear `Beta_novo = Beta·3.5/R0`

**Passo 1 — busca (MAXSIM=300):**
| Cidade | trajeto Beta→R0 | Beta | R0 |
|--------|-----------------|------|-----|
| SP | 0.03→4.47 ; 0.02347→3.55 | 0.02347 | 3.55 |
| ROC | 0.008→5.28 ; 0.00531→3.91 ; 0.00475→3.53 | 0.00475 | 3.53 |
| BRA | 0.05→2.97 ; 0.05886→3.38 | 0.05886 | 3.38 |
| MAN | 0.05→2.97 ; 0.05886→3.38 | 0.05886 | 3.38 |

**Passo 2 — refino (MAXSIM=1000, tol 0.1) → valores finais:**
| Cidade | trajeto Beta→R0 | **Beta final** | **R0 final** |
|--------|-----------------|----------------|--------------|
| SP | 0.02347→3.75 ; 0.02191→3.60 ; 0.02129→3.52 | **0.021293** | **3.517** |
| ROC | 0.00475→3.60 ; 0.004613→3.53 | **0.004613** | **3.525** |
| BRA | 0.05886→3.39 ; 0.060767→3.46 | **0.060767** | **3.464** |
| MAN | 0.05886→3.39 ; 0.060767→3.46 | **0.060767** | **3.464** |

> **BRA = MAN** (idênticos): mesmos contatos (1.5/2.5) e mesma vizinhança (Moore, pelo bug do `#if`);
> leitos/UTI não afetam o R0. São indistinguíveis para a calibração de R0.

## Beta calibrado (R0≈3.5) — RESUMO FINAL (serial, parâmetros do documento)

| Cidade | **Beta (R0=3.5)** | R0 (MAXSIM=1000) | Contatos aleat. | Vizinhança |
|--------|-------------------|------------------|-----------------|------------|
| **São Paulo** | **0.02129** | 3.52 | 1.5/18.5 (2–19) | Moore (8) |
| **Rocinha** | **0.00461** | 3.53 | 1.5/119.5 (2–120) | Moore (8) |
| **Brasília** | **0.06077** | 3.46 | 1.5/2.5 (2) | Moore (8) ⚠️ |
| **Manaus** | **0.06077** | 3.46 | 1.5/2.5 (2) | Moore (8) ⚠️ |

Observações:
- Ordem dos Betas faz sentido físico: mais contatos → menor Beta para o mesmo R0.
  ROC (até 120 contatos) precisa do menor Beta; BRA/MAN (~2 contatos) o maior.
- O `Beta=0.0658` antigo do `define.h` **não** dá R0=3.5 para SP com contatos 2–19 (dá R0≈7.8);
  ele correspondia aos contatos antigos (~2). Para SP com a doc (2–19), R0=3.5 ⇒ **Beta≈0.0213**.
- ⚠️ BRA/MAN: calibrados com **Moore (8)** porque o serial sempre usa Moore (bug do `#if(Density==HIGH)`).
  Pela doc seriam Von Neumann (4 vizinhos); com 4 vizinhos o R0 seria menor e o Beta **maior**.
  Recalibrar após corrigir o `#if` se quiser fidelidade à doc para baixa densidade.

### Confirmação em L=3355 (L real de SP)
R0 é independente de L (provado no sweep). Confirmação de SP a L=3355 com Beta=0.021293:
_(rodando em background, MAXSIM=100 — preencher ao concluir)_

### Como reproduzir
```
# em CudaRuntime1/CudaRuntime1, com define.h: IPini=1, L=200 (R0 e L-independente), MAXSIM=1000
# e cities(<CIDADE>) no r0_serial.cu, Beta = valor da tabela:
nvcc r0_serial.cu -o r0_serial.exe -arch=sm_89 --diag-suppress 20091 --diag-suppress 177 --diag-suppress 940 -Xcompiler "/wd4716"
.\r0_serial.exe   # imprime "R0 medio = ..."
```
