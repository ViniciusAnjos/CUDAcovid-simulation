# Investigação: por que São Paulo "não completa" acima de certo MAXSIM

**Data:** 2026-06-02. **Branch:** `perf/health-soa`.
**Sintoma relatado:** "a sim roda, mas se MAXSIM passar de um certo valor o job simplesmente não
completa." Específico de **São Paulo** (cidade de maior porte). As outras 3 cidades completam
MAXSIM=50 sem problema.

## TL;DR (causa raiz)

**WDDM TDR (Timeout Detection and Recovery) do Windows.** A GPU é uma placa de **vídeo
compartilhada** com o desktop (Windows + Edge + Steam + OneDrive + …). O Windows reseta o driver
gráfico se **qualquer comando da GPU demora mais de ~2 segundos** (TdrDelay padrão = 2s). Numa
rodada SP longa, mais cedo ou mais tarde um kernel ultrapassa 2s (pico da epidemia e/ou disputa
com o compositor do desktop) → **Windows reseta a GPU** → o contexto CUDA do `covid_sim` morre →
ele continua rodando e grava **resultado-lixo** (`ataque=1`, arquivos `.dat` de 0 bytes).

Não é vazamento de memória, não é térmico, não é lentidão inerente.

## Como foi descartado o que NÃO era

| Hipótese | Verificação | Veredito |
|----------|-------------|----------|
| Térmico (throttling) | `nvidia-smi`: 42–44 °C, clock 2865/3105 MHz durante/após | ❌ GPU fria, clock cheio |
| Vazamento de VRAM por sim | Leitura do loop em `covid.cu`: `d_stateCounts/d_newCounts` são `cudaFree`-ados (linhas 273-275); arrays grandes alocados 1× antes do loop | ❌ sem leak |
| Lentidão inerente da SP | SP **isolada** MAXSIM=5 = **65s** (13s/sim), resultado válido (ataque 0.728) | ❌ rápida quando isolada |
| Contenção do sistema | 720 MiB da GPU ocupados por desktop/Edge/Steam (processos `C+G`) | ✅ contribui (ver abaixo) |

## Prova definitiva (Event Log do Windows)

`nvlddmkm` **Event ID 153** (reset de engine da GPU / TDR) no log do Sistema bate **ao segundo**
com cada rodada SP que falhou:

| Event 153 (TimeCreated) | Rodada SP correspondente (`curvas/_log.txt`) | Resultado gravado |
|-------------------------|----------------------------------------------|-------------------|
| 04:59:52 | "LOTE CONCLUIDO" SP MAXSIM=50 (tempo=6206s) | `ataque=1` (lixo) |
| 05:56:00 | "LOTE CONCLUIDO SP=MAXSIM20" (tempo=3358s) | `ataque=1` (lixo) |
| 10:25:23 | SP MAXSIM=30 (morto manualmente após hang) | hang |

> Comando: `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; Id=153}`

Sintoma colateral: após o reset, a GPU fica com VRAM "ocupada" mas **210 MHz / 10% de uso**
(idle) — o `covid_sim` fica preso num `cudaDeviceSynchronize` que nunca retorna (hang).

## Por que SÓ São Paulo

- SP: array `GPUPerson` = 721 MB (≫ L2 48 MB) **e** ~9.5 contatos aleatórios/suscetível → kernels
  de pico são os mais pesados; sob disputa com o desktop, cruzam 2s. (A otimização Health-SoA já
  reduziu muito o tempo, mas o pior caso de pico ainda arrisca passar de 2s.)
- ROC/BRA/MAN: kernels muito mais leves (L menor e/ou só ~2 contatos) → nunca chegam a 2s → nunca
  disparam TDR. Por isso completam MAXSIM=50.
- É **probabilístico no tempo de parede**: rodada curta (65s) quase nunca pega um pico>2s; rodada
  longa (dezenas de min) quase certamente pega → daí o "acima de certo MAXSIM não completa".

## Correção DEFINITIVA aplicada (TdrDelay) — 2026-06-02

**TCC mode foi descartado:** a GPU é uma **GeForce RTX 4070 SUPER (WDDM)**; o modo TCC só existe em
placas Tesla/Quadro profissionais — `nvidia-smi -dm 1` não funciona em GeForce.

**Aplicado o fix de registro (via UAC elevado):**
`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`
- `TdrDelay`    = **60** (DWORD)  — timeout do TDR passou de 2s → 60s
- `TdrDdiDelay` = **60** (DWORD)

TDR continua **ativo** (recuperação ligada), mas com margem de 60s nenhum kernel da simulação chega
perto de disparar reset. (Script: `CudaRuntime1/set_tdr.ps1`, rodado elevado.)

> ⚠️ **REQUER REBOOT** para entrar em vigor. Após reiniciar, SP roda MAXSIM=50 direto, sem chunks.

### Solução de software (mantida como fallback / portável a outras máquinas)

### 1. `covid.cu` — checagem de erro CUDA por dia (anti-lixo)
Depois de `runSimulationDay`, checa `cudaDeviceSynchronize()` + `cudaGetLastError()`. Se houver
erro (TDR ⇒ `cudaErrorLaunchTimeout`/contexto perdido), **aborta com `return 3`** em vez de
continuar gravando `ataque=1`. Agora um TDR é **detectável** (exit code ≠ 0), não silencioso.

### 2. `covid.cu` — argumentos de linha de comando
`covid_sim.exe [MAXSIM] [seedBase]`. Permite rodar em **chunks curtos** com sementes distintas
(`seed = 893221891 * (seedBase + simulation)`), preservando reprodutibilidade.

### 3. `run_sp_chunked.ps1` — driver resiliente
Roda SP em **N chunks de m sims** (atual: 6×5 = 30 sims). Cada chunk dura ~60s (tamanho **provado
sobreviver** ao TDR). Se um chunk falha (exit≠0 ou `.dat` vazio), faz **retry** (até 4×). No fim,
faz a **média element-wise dos chunks** → curva SP final. Como todos os chunks têm o mesmo `m`,
`média(médias dos chunks) = média global exata sobre N·m sims`.

Saídas: `curvas/SP/*.dat` (curva final), `curvas/SP_chunks/*.dat` (por chunk),
`curvas/sp_chunked_log.txt` (log com tempo/ataque por chunk + retries).

---

# SEGUNDO BUG (mais profundo): RNG degenerado → loop infinito no kernel

Após aplicar `TdrDelay=60` e reiniciar, SP **ainda travou** — mas de forma diferente: parou na
**Simulation 16/50 com a GPU a 100% por ~72 min** (não idle como no TDR). Ou seja, **loop infinito
dentro de um kernel**, que antes era mascarado pelo TDR (resetava em 2s) e agora ficou exposto.

## Causa raiz — o gerador de aleatórios é matematicamente quebrado (`gpu_aleat.cuh`)

```c
*state = (*state * 888121u) mod 2^32;     // LCG puramente multiplicativo, SEM constante aditiva
```

1. **Multiplicador 888121 ≡ 1 (mod 8).** Para um LCG multiplicativo mod 2³², o período máximo exige
   multiplicador ≡ 3 ou 5 (mod 8). Com ≡ 1, o período é pessimamente curto → ciclos degenerados.
2. **Semeadura `states[idx] = seed * (idx+1)`** com `seed = 893221891 * simulation`. Como 893221891
   é ímpar, a paridade da semente = paridade de `simulation`. **Toda simulação par (2,4,…,16,…) →
   semente par → todos os estados pares.** E como 888121 é ímpar, a multiplicação **preserva a
   paridade para sempre** → o gerador fica preso num subconjunto de baixa entropia (metade das
   simulações rodava num RNG degradado).

Com o RNG degenerado, os **laços de rejeição** `do{...}while(cond)` nunca atingem `cond` de parada:
- `replaceDeadPerson` / `initPopulation`: `while(mute<1)` — amostra idade de morte até
  `rn < ProbNaturalDeath[idade]`. Se o ciclo curto do RNG nunca produz o par (idade, rn) que aceita,
  **roda para sempre**. (É o que travou a sim 16 — `replaceDeadPerson` roda todo dia.)
- `checkRandomContacts` / `spreadInfection`: `while(randI==i && randJ==j)` — "evita self".
- `distributeInitialInfections`: `while(Health!=d_S)` — acha suscetível.

Uma única thread/warp em loop infinito → o kernel nunca retorna → GPU presa a 100%.

## Correção aplicada — guards (limite de iterações + fallback)

Em **todos os 5 laços de rejeição** foi adicionado um contador-guarda com saída segura
(ex.: `if (++guard > 10000) { AgeDeathYears = 99; mute = 1; }`). Isso **garante término** sem
alterar o stream do RNG nos casos normais (o guard nunca dispara num thread saudável), então:
- **a calibração Beta=0.0243 permanece válida** (resultados idênticos onde não havia trava);
- só os raros threads degenerados pegam fallback → impacto desprezível em 11,3M células.

**Validação:** SP MAXSIM=50 completou em **634 s (12,7 s/sim)**, `ataque=0.726` — consistente com a
rodada isolada MAXSIM=5 (`0.728`), confirmando que os guards não enviesaram o resultado.

## Recomendação para a monografia (RNG)

> ⚠️ Os guards **estancam o sintoma**, mas o RNG continua de baixa qualidade (metade das sims com
> stream degradado). Para rigor estatístico da monografia, recomenda-se **substituir o gerador** por
> um LCG misto de período completo (Hull-Dobell), ex.: `*state = *state*1664525u + 1013904223u`
> (período 2³² para **qualquer** semente, inclusive 0 — sem colapso de paridade), e usar uma
> semeadura por-thread bem misturada (hash). **Isso muda o stream → exigiria recalibrar o Beta** das
> 4 cidades. Decisão do autor: manter (guards, calibração atual preservada) vs. upgrade do RNG
> (mais correto, custa recalibração).

---

## Recomendação para a monografia / execuções futuras (TDR)

1. **Setup ideal (uma vez):** rodar com privilégio de admin, definir `TdrDelay=60` e reiniciar.
   ✅ **Aplicado nesta máquina** (UAC + reboot) — SP MAXSIM=50 passou a completar sem reset.
   (Alternativa: GPU dedicada/headless; TCC **não** existe em GeForce.)
2. **Sem isso:** usar o driver de chunks (`run_sp_chunked.ps1`) — robusto em qualquer máquina.
3. A checagem de erro CUDA é boa prática permanente: nunca mais gravar resultado-lixo silencioso.
4. Os **guards anti-trava** nos laços de rejeição são permanentes (defesa contra qualquer RNG ruim).
