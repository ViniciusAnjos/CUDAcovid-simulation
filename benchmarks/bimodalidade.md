# Por que a epidemia é bimodal (e por que MAXSIM precisa ser ≥ 30)

Descoberto ao validar serial × GPU: o aparente "3× de divergência" era **ruído de amostra pequena
(MAXSIM=5)** porque a epidemia, começando de poucos casos, é **bimodal**. Não é bug — é a natureza
estocástica de epidemias.

## O que é

Rodando a simulação muitas vezes e olhando a **taxa de ataque final de cada rodada**, não há um pico
central — há **dois picos**:

```
nº de rodadas
   │ █                                █
   │ █ █                            █ █
   │ █ █ █                        █ █ █
   └──────────────────────────────────────  taxa de ataque
     ~0% (fracasso/fade-out)        ~80% (surto grande)
```

Cada rodada **ou fracassa (~0%) ou explode (~80%)** — quase nada no meio.

## Por quê

A epidemia começa com **5 IP** (pacientes-zero). O destino é decidido na **fase inicial**, com
pouquíssimos infectados, onde o **acaso domina**:

- **Fade-out (extinção estocástica):** por azar, os poucos infectados iniciais se recuperam/transicionam
  antes de infectar o suficiente. A cadeia morre → ~0%.
- **Takeoff (surto maior):** a transmissão se estabelece; aí a lei dos grandes números assume e a
  epidemia cresce quase deterministicamente até saturar → ~80%.

Não há meio-termo estável: ou estabelece e vai até o fim, ou apaga no começo.

### Intuição (processo de ramificação)
Cada infectado gera em média R0=3.5 secundários, mas é um **número aleatório**. No começo, com poucos
casos, há chance real de todas as cadeias darem 0 por azar. Fatores que deixam ESTE modelo sensível
ao fade-out inicial:
- **Só 5 sementes** (poucas cadeias independentes).
- **Latência longa (E ~13 dias)** — demora a engrenar, muito tempo para azar.
- **`StateTime` pode ser 0** — um IP sorteado com duração 0 quase não espalha (semente desperdiçada).
- **Estrutura espacial** — no início os infectados saturam localmente os vizinhos, reduzindo o R
  efetivo logo no começo.

## Por que quebrou o MAXSIM=5

Com poucas rodadas a **média é instável** — depende de quantas das poucas decolaram:

> média ≈ (fração que decola) × (tamanho final ~80%)

| MAXSIM | Serial (β=0.0213) | GPU (β=0.0213) |
|--------|-------------------|----------------|
| 5 | **21%** (azar: ~1-2 de 5 decolaram) | 66% |
| 30 | **70.4%** (≈ 0.88 × 0.80) | 66.3% |

Serial deu azar com 5 sims (poucas decolagens → 21%); GPU por acaso teve mais (66%). Daí a falsa
impressão de "3×". Com 30 sims, **serial ≈ GPU** (70% vs 66%).

## Lição prática (para a monografia)

- **Não é defeito** — é estocasticidade de epidemias iniciadas de poucos casos (dicotomia clássica
  "surto menor vs surto maior", teoria de processos de ramificação).
- A média sobre **MAXSIM=5 ou 10 é estatisticamente não-confiável** para comparar serial × GPU.
- **Usar MAXSIM ≥ 30** (idealmente mais) para médias estáveis e comparáveis.
- Alternativa de análise: reportar separadamente a **probabilidade de takeoff** e o **tamanho final
  condicional aos takeoffs** (mais informativo que a média crua num regime bimodal).
