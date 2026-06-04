# Otimização de desempenho: array compacto de Health (SoA) — para a monografia

Branch: `perf/health-soa`. Otimização de **layout de memória** que acelera a cidade de maior porte
(São Paulo) em ~12–16×, mantendo os resultados **idênticos**.

## 1. O problema (diagnóstico)

A simulação cresceu para o L real de cada cidade. São Paulo (L=3355) ficou **ordens de magnitude**
mais lento que as outras na GPU (RTX 4070 SUPER), apesar do mesmo algoritmo:

| Cidade | L | Array de população¹ | Cabe no L2 (48 MB)? | Contatos aleat./suscetível | Tempo/sim |
|--------|------|---------------------|---------------------|----------------------------|-----------|
| Rocinha | 264 | ~4.5 MB | ✅ sim | ~60 | rápido |
| Manaus | 1343 | ~116 MB | ❌ não | ~2 | ~2.8 s |
| Brasília | 1604 | ~165 MB | ❌ não | ~2 | ~2.7 s |
| **São Paulo** | 3355 | **~721 MB** | ❌ não (15×) | ~9.5 | **~130–190 s** |

¹ `GPUPerson` = 13 ints, `__align__(16)` = 64 bytes/célula; array = (L+2)² células.

**Causa raiz — acesso aleatório fora do cache.** Os mecanismos de contágio leem o campo `Health` de
células **sorteadas aleatoriamente** (contatos aleatórios) em toda a malha:
- `checkRandomContacts` (suscetível-driven): ~9.5 leituras aleatórias por suscetível/dia.
- `spreadInfection` (infected-driven): idem para cada infeccioso.

Cada leitura aleatória vai a um endereço imprevisível do array de 721 MB. Como 721 MB ≫ L2 (48 MB),
**toda leitura é cache miss → latência de DRAM (~centenas de ciclos)**. Em SP isso são ~10⁸ leituras
aleatórias/dia × 400 dias × MAXSIM → bilhões de acessos à DRAM = o gargalo.

Por que as outras cidades não sofrem:
- **Rocinha:** array de 4.5 MB **cabe no L2** → acesso aleatório rápido (mesmo com ~60 contatos).
- **Brasília/Manaus:** não cabem, mas só ~2 contatos → volume de acesso aleatório baixo.
- **São Paulo:** o pior caso — array gigante (fora do cache) **E** muitos contatos.

## 2. A solução

O acesso aleatório só precisa do **`Health`** da célula-alvo (1 valor), não dos 64 bytes do
`GPUPerson`. Mantemos um **array compacto paralelo** só com Health (Structure-of-Arrays para o campo
quente):

```
__device__ unsigned char* d_HealthC;   // 1 byte/célula (Health ∈ 1..14)
```

- Tamanho para SP: 11.3M × 1 byte = **~11 MB → cabe folgado no L2 (48 MB)**.
- (`int` daria 45 MB, apertado demais p/ um cache de 48 MB compartilhado; `unsigned char` resolve.)

Os acessos aleatórios passam a ler de `d_HealthC` → **cache hit** em vez de DRAM.

## 3. Implementação (localizada)

- `gpu_define.cuh`: declara `__device__ unsigned char* d_HealthC;`.
- `update_kernel.cuh`: ao finalizar o `Health` do dia, escreve `d_HealthC[idx] = Health`
  (mantém sincronizado, custo zero — reaproveita o `finalState` já calculado). Kernel
  `syncHealthC_kernel` para o sync inicial (após semear, antes do dia 1).
- `gpu_neighbors.cuh`: os 3 sites de leitura aleatória de Health passam a ler `d_HealthC[idx]`
  (`checkLocalContacts`, `checkRandomContacts`, `spreadInfection`).
- `covid.cu`: aloca `d_HealthC_buf` (gridSize bytes), seta o símbolo, sync inicial, libera no fim.

O array completo `GPUPerson` (721 MB) continua na DRAM, mas é acessado de forma **coalescida** (cada
thread lê a própria célula) — sem problema. Só o acesso aleatório passou a ser compacto.

## 4. Validação (resultados idênticos)

Como `d_HealthC` espelha exatamente `Health` (mesmos valores, sem RNG extra, mesma lógica), o
resultado é preservado. Verificado em **Manaus** (config idêntica à versão não-otimizada):

| | Ataque (dia 400) | Tempo |
|---|---|---|
| Não-otimizado | 0.7591 | 142 s |
| Otimizado | **0.7591** (idêntico) | 143 s |

Manaus não acelera (só ~2 contatos, não era memory-bound) — confirma que a mudança não tem efeito
colateral onde não é necessária.

## 5. Speedup (São Paulo)

| | Tempo/sim (SP, L=3355) | Speedup |
|---|------------------------|---------|
| Não-otimizado | ~130–190 s | 1× |
| **Otimizado** | **~11.8 s** | **~12–16×** |

MAXSIM=50 de SP: de **~1.5–2 h** para **~10 min**. (resultado final SP: ver `epidemia_cidades.md`.)

## 6. Lição (para a monografia)

Em ABM espacial na GPU, o gargalo da cidade grande **não é computação, é latência de memória** no
acesso aleatório. A otimização clássica é **separar o campo quente num array compacto (SoA)** para
caber no cache — ganho de ~uma ordem de magnitude no caso memory-bound, sem alterar resultados.
É um exemplo didático de otimização guiada por análise de cache (working set vs tamanho do L2).
