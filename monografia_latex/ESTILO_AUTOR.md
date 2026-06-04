# Perfil de Estilo do Autor — Vinícius Santos Anjos da Silva

> Guia de voz para escrever a monografia (LaTeX/abnTeX2) emulando o estilo do autor.
> Base: análise do anteprojeto `Projeto_Monografico_Vinicius_Anjos` (main.tex, 25/11/2022).
>
> **Regra geral:** emular a VOZ em ~70% (trejeitos de estilo abaixo), CORRIGIR 100% os
> erros ortográficos/gramaticais (vícios abaixo). Banca penaliza os erros; a voz é o que
> torna o texto autêntico.
>
> **Calibragem (~70%, não cópia literal):**
> - MANTER: conectivos-assinatura, abertura por finalidade, aposto definidor, voz passiva
>   com "se", topicalização por preposição, termos de hardware em inglês.
> - SUAVIZAR: nem todo período precisa ter 4–5 linhas — quebrar frases longas demais
>   quando a clareza pedir; não empilhar conectivo sobre conectivo no mesmo parágrafo;
>   variar o léxico (não repetir "diversos/diversas" em excesso).
> - O texto deve soar reconhecidamente como o autor, porém um pouco mais claro e polido
>   que o anteprojeto.

---

## Trejeitos de estilo — MANTER

1. **Períodos longos encadeados por vírgula.** Padrão de 3–5 linhas; 3–4 orações
   emendadas até fechar a ideia. Ponto final é raro dentro de um parágrafo.

2. **Abertura por circunstância/finalidade**, não pelo sujeito:
   *"Para simular uma epidemia...", "Com o intuito de reduzir...", "Pelas características
   do vírus...", "Ao desenvolver programas..."*

3. **Conectivos-assinatura** (usar com frequência):
   `no qual` · `tal que` · `dentre eles` · `entretanto` · `logo` (= portanto) ·
   `além de` · `através de` · `em detrimento de` · `uma vez que` ·
   `com o intuito de` · `de acordo com` · `assim`

4. **Voz passiva sintética com "se" + nominalizações:**
   *"utiliza-se de modelos", "são utilizados", "é dada por", "torna-se"*;
   substantivos de ação: *a paralelização, a calibração, a implementação, o espalhamento*.

5. **Definição em aposto** — todo termo técnico entra com sigla/tradução entre
   parênteses + aposto explicativo:
   *"Suscetíveis (S), representados por aqueles que...", "API (Application Programming
   Interface)", "R_0, parâmetro que representa quantos agentes..."*

6. **Topicalização por preposição** — fixar o domínio no começo da frase:
   *"Em ABM...", "Em CUDA...", "No modelo...", "No host...", "Na GPU..."*

7. **Termos de hardware em itálico inglês**, sem traduzir:
   *throughput, cores, warps, host, device, streaming multiprocessors*.

8. **Referência explícita a elementos:** *"A Figura X representa...", "A Tabela Y mostra..."*

9. **Léxico recorrente:** *diversos/diversas* (características, parâmetros, tipos),
   *indivíduo/agente, rede, vizinhança, contágio, espalhamento*.

10. **Listas com `itemize`** para enumerar tipos/objetivos/condições.

---

## Vícios recorrentes — CORRIGIR (não reproduzir no texto final)

| Vício no anteprojeto | Forma correta |
|----------------------|---------------|
| parãmetros (til por circunflexo) | parâmetros |
| extende | estende |
| poulação | população |
| oara | para |
| progamação / progamador | programação / programador |
| abragem | abrangem |
| Von Neumman | Von Neumann |
| Brasilía | Brasília |
| índividuos | indivíduos |
| Vínícius (acento duplo) | Vinícius |
| Vírgula entre sujeito e verbo: "O vírus..., se espalha" | sem vírgula |
| Comma splice (vírgula no lugar de ponto/ponto-e-vírgula) | pontuar corretamente |
| Repetição próxima ("...em tal região. ...em tal região") | variar/eliminar |

---

## Exemplo-referência (voz emulada, já corrigida)

> "Para paralelizar o modelo baseado em agentes em CUDA, a rede que representa a
> população é mapeada de modo que cada agente seja associado a uma thread, no qual o
> estado de saúde de cada indivíduo é atualizado de forma independente a cada passo de
> tempo. Com o intuito de evitar a divergência entre as warps, o lado da rede é ajustado
> para um múltiplo de 32, uma vez que threads de uma mesma warp executam a mesma
> instrução. Diferentemente da implementação serial, na qual os agentes são percorridos
> sequencialmente por um único laço, na GPU os kernels realizam a varredura de toda a
> rede simultaneamente, e logo o tempo de simulação passa a depender da largura de banda
> de memória do device em detrimento do número de operações."

Elementos presentes: abertura por finalidade · `no qual`/`uma vez que`/`em detrimento de`/`logo`
· aposto definidor · topicalização "na GPU..." · termos *thread/warp/device* em inglês.
