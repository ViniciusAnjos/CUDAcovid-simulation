# Variabilidade entre simulações (estocasticidade do modelo)

Cada cidade rodada com **50 simulações independentes** (GPU, β calibrado R0=3,5, L real, 400 dias),
guardando o bruto de cada simulação (`prevalence_N.dat`). Objetivo: medir a incerteza em torno das
médias usadas nos resultados.

## Robustez no regime calibrado (R0 = 3,5)

Taxa de ataque final (1 − S no dia 400), sobre as 50 simulações:

| Cidade | Média | Desvio padrão | Mínimo | Máximo |
|--------|-------|---------------|--------|--------|
| Rocinha | 58,8% | **0,5 p.p.** | 57,6% | 59,8% |
| Manaus | 75,9% | **0,1 p.p.** | 75,5% | 76,1% |
| Brasília | 76,7% | **0,1 p.p.** | 76,4% | 76,9% |
| São Paulo | 72,6% | **0,9 p.p.** | 68,5% | 73,2% |

**Conclusão:** no β calibrado (R0=3,5, bem acima do limiar epidêmico) os resultados são **altamente
robustos** — todas as 50 simulações decolam e convergem para praticamente o mesmo ataque (desvio
≤ 0,9 p.p.). **Não há bimodalidade** aqui: as médias de MAXSIM=50 são estatisticamente confiáveis.
A maior variabilidade aparece no **timing** da epidemia (banda mais larga no meio da curva de S(t)),
não no desfecho final — ver `graficos/variabilidade/bandas_S.png` e `bandas_infecciosos.png`.

## Incerteza perto do limiar epidêmico

Reduzindo β em direção ao limiar (modelo espacial em lattice), a estocasticidade cresce: algumas
simulações **se extinguem** (extinção estocástica) e as demais crescem em graus variados — mas **sem
dois picos nítidos** (a bimodalidade clássica de campo-médio é suavizada pela estrutura espacial).

Exemplo (Rocinha, 50 sims):

| β | Média do ataque | Desvio | Extinções (<10%) |
|------|-----------------|--------|-------------------|
| 0,0030 | 8% | — | 36/50 |
| 0,0034 | 17,9% | **3,6 p.p.** | 2/50 (espalha 0–22%) |
| 0,0042 | 43,6% | — | 1/50 |
| **0,0049 (R0=3,5)** | **58,8%** | **0,5 p.p.** | 0/50 |

→ Perto do limiar o desvio é **~7× maior** (3,6 vs 0,5 p.p.). Ver
`graficos/variabilidade/robustez_vs_limiar.png` (histogramas lado a lado).

**Implicação:** a calibração para R0=3,5 não só dá o R0-alvo como coloca a simulação num regime
**robusto/reprodutível** — longe da zona de alta incerteza próxima ao limiar.

## Reprodução
`CudaRuntime1/run_variabilidade.ps1` roda as 4 cidades MAXSIM=50 salvando os brutos em
`curvas/variabilidade/<cidade>/`; `curvas/plot_variabilidade.py` gera as figuras. (Os 50 brutos por
cidade são regeneráveis e não versionados.)
