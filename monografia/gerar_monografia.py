# -*- coding: utf-8 -*-
"""Gera a monografia (ABNT) em .docx usando python-docx."""
import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

G = r"C:\Users\User\source\repos\CudaRuntime1-gpu\benchmarks\curvas\graficos"
OUT = r"C:\Users\User\source\repos\CudaRuntime1-gpu\monografia\Monografia_COVID_CUDA_ABNT.docx"

doc = Document()

# ---------------- configuração de página e estilo (ABNT) ----------------
sec = doc.sections[0]
sec.page_height = Cm(29.7); sec.page_width = Cm(21.0)          # A4
sec.top_margin = Cm(3); sec.left_margin = Cm(3)               # ABNT: sup/esq 3 cm
sec.bottom_margin = Cm(2); sec.right_margin = Cm(2)           # inf/dir 2 cm

normal = doc.styles['Normal']
normal.font.name = 'Times New Roman'; normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
pf = normal.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.space_after = Pt(0); pf.space_before = Pt(0)

# estilos de titulo (ABNT: Times 12 negrito)
for i, nome in enumerate(['Heading 1', 'Heading 2', 'Heading 3'], start=1):
    st = doc.styles[nome]
    st.font.name = 'Times New Roman'; st.font.size = Pt(12); st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(18); st.paragraph_format.space_after = Pt(12)
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    st.paragraph_format.keep_with_next = True

# numero de pagina no cabecalho (canto direito)
def add_page_number(sec):
    p = sec.header.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run(); r.font.name = 'Times New Roman'; r.font.size = Pt(10)
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'PAGE'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'end')
    r._r.append(f1); r._r.append(it); r._r.append(f2)
add_page_number(sec)

# ---------------- helpers ----------------
def corpo(texto, indent=True):
    p = doc.add_paragraph(texto)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    return p

def centro(texto, bold=False, size=12, caps=False):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(texto.upper() if caps else texto)
    r.bold = bold; r.font.size = Pt(size); r.font.name = 'Times New Roman'
    return p

def h1(texto):
    doc.add_heading(texto, level=1)
def h2(texto):
    doc.add_heading(texto, level=2)

_figN = [0]
def figura(titulo, path, fonte="Elaborado pelo autor (2026).", largura=15.0):
    _figN[0] += 1
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figura {_figN[0]} – {titulo}"); r.font.size = Pt(11)
    if os.path.exists(path):
        pim = doc.add_paragraph(); pim.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pim.add_run().add_picture(path, width=Cm(largura))
    src = doc.add_paragraph(); src.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = src.add_run("Fonte: " + fonte); rs.font.size = Pt(10)
    doc.add_paragraph()

_tabN = [0]
def tabela(titulo, headers, linhas, fonte="Elaborado pelo autor (2026)."):
    _tabN[0] += 1
    cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Tabela {_tabN[0]} – {titulo}"); r.font.size = Pt(11)
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for j, htxt in enumerate(headers):
        hdr[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = hdr[j].paragraphs[0].add_run(htxt); rr.bold = True; rr.font.size = Pt(11)
    for lin in linhas:
        cells = t.add_row().cells
        for j, val in enumerate(lin):
            cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = cells[j].paragraphs[0].add_run(str(val)); rr.font.size = Pt(11)
    # bordas ABNT: topo e base apenas
    tbl = t._tbl
    src = doc.add_paragraph(); src.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rs = src.add_run("Fonte: " + fonte); rs.font.size = Pt(10)
    doc.add_paragraph()

def toc():
    p = doc.add_paragraph(); r = p.add_run()
    f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
    it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = 'TOC \\o "1-3" \\h \\z \\u'
    f2 = OxmlElement('w:fldChar'); f2.set(qn('w:fldCharType'), 'separate')
    tt = OxmlElement('w:t'); tt.text = "Abra no Word e atualize o sumario (clique e tecle F9)."
    f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
    r._r.append(f1); r._r.append(it); r._r.append(f2); r._r.append(tt); r._r.append(f3)

# ============================================================ CAPA
for _ in range(1): doc.add_paragraph()
centro("UNIVERSIDADE FEDERAL FLUMINENSE", bold=True)
centro("INSTITUTO DE CIÊNCIAS EXATAS – CAMPUS VOLTA REDONDA", bold=True)
for _ in range(6): doc.add_paragraph()
centro("VINÍCIUS SANTOS ANJOS DA SILVA", bold=True)
for _ in range(6): doc.add_paragraph()
centro("PARALELIZAÇÃO EM CUDA DE UMA SIMULAÇÃO EPIDEMIOLÓGICA BASEADA EM AGENTES DE COVID-19 EM ÁREAS URBANAS DO BRASIL", bold=True, size=14)
for _ in range(10): doc.add_paragraph()
centro("Volta Redonda – RJ", bold=True)
centro("2026", bold=True)
doc.add_page_break()

# ============================================================ FOLHA DE ROSTO
for _ in range(1): doc.add_paragraph()
centro("VINÍCIUS SANTOS ANJOS DA SILVA", bold=True)
for _ in range(6): doc.add_paragraph()
centro("PARALELIZAÇÃO EM CUDA DE UMA SIMULAÇÃO EPIDEMIOLÓGICA BASEADA EM AGENTES DE COVID-19 EM ÁREAS URBANAS DO BRASIL", bold=True, size=14)
for _ in range(3): doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.left_indent = Cm(8); p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
r = p.add_run("Monografia apresentada ao Curso de Graduação da Universidade Federal Fluminense, "
              "Campus Volta Redonda, como requisito parcial para a obtenção do grau de Bacharel.")
r.font.size = Pt(11)
for _ in range(3): doc.add_paragraph()
p2 = doc.add_paragraph(); p2.paragraph_format.left_indent = Cm(8)
r2 = p2.add_run("Orientador: Prof. Dr. Aquino Lauri de Espíndola"); r2.font.size = Pt(11)
for _ in range(8): doc.add_paragraph()
centro("Volta Redonda – RJ", bold=True)
centro("2026", bold=True)
doc.add_page_break()

# ============================================================ RESUMO
centro("RESUMO", bold=True); doc.add_paragraph()
corpo("Modelos epidemiológicos baseados em agentes (ABM) reproduzem a propagação de doenças com alto "
      "detalhe espacial e individual, mas têm custo computacional elevado, o que limita o tamanho das "
      "populações simuladas e o número de réplicas estatísticas. Este trabalho apresenta a paralelização "
      "em CUDA de uma simulação ABM do tipo SEIR estendido de COVID-19, aplicada a quatro áreas urbanas "
      "brasileiras (São Paulo, Rocinha, Brasília e Manaus), com o objetivo de validar que a implementação "
      "em GPU reproduz fielmente a versão serial de referência e de medir o ganho de desempenho. A "
      "validação foi conduzida comparando-se as curvas epidêmicas das duas implementações sob o mesmo "
      "parâmetro de transmissão. O diagnóstico da divergência inicial, por bissecção empírica dos "
      "mecanismos, revelou que a discrepância era causada por dois defeitos na implementação serial — "
      "e não na GPU: um isolamento social ativado por engano por uma diretiva de pré-processador e uma "
      "sobrescrita indevida que apagava infecções. Após as correções, serial e GPU produzem a mesma "
      "epidemia (erro quadrático médio da fração de suscetíveis inferior a 1,2 ponto percentual nas quatro "
      "cidades). O ganho de desempenho (speedup) variou de 1,8 vez (Rocinha) a 56 vezes (São Paulo), "
      "crescendo com o tamanho do reticulado. A análise de hardware — perfilamento por núcleo, "
      "invariância ao tamanho de bloco e comparação entre duas GPUs distintas — demonstrou que a "
      "simulação é limitada pela largura de banda de memória, o que orientou uma otimização de layout de "
      "dados (Structure-of-Arrays para o campo de saúde) que acelerou o maior caso em ordem de grandeza.")
corpo("Palavras-chave: Computação paralela. CUDA. Modelagem baseada em agentes. Epidemiologia "
      "computacional. COVID-19.", indent=False)
doc.add_page_break()

# ============================================================ ABSTRACT
centro("ABSTRACT", bold=True); doc.add_paragraph()
corpo("Agent-based epidemiological models (ABM) reproduce disease spread with high spatial and individual "
      "detail, but at a high computational cost that limits both the population size and the number of "
      "statistical replicates. This work presents the CUDA parallelization of an extended SEIR ABM "
      "simulation of COVID-19 applied to four Brazilian urban areas (São Paulo, Rocinha, Brasília and "
      "Manaus), aiming to validate that the GPU implementation faithfully reproduces the reference serial "
      "version and to measure the performance gain. Validation compared the epidemic curves of both "
      "implementations under the same transmission parameter. An empirical mechanism-by-mechanism "
      "bisection revealed that the initial divergence was caused by two defects in the serial code — "
      "not in the GPU: a social-isolation policy accidentally enabled by a preprocessor directive, and an "
      "improper overwrite that erased infections. After the fixes, serial and GPU produce the same epidemic "
      "(root-mean-square error of the susceptible fraction below 1.2 percentage points across all four "
      "cities). The speedup ranged from 1.8x (Rocinha) to 56x (São Paulo), growing with the lattice size. "
      "Hardware analysis — per-kernel profiling, block-size invariance and a two-GPU comparison — "
      "showed that the simulation is memory-bandwidth bound, which guided a data-layout optimization "
      "(Structure-of-Arrays for the health field) that sped up the largest case by an order of magnitude.")
corpo("Keywords: Parallel computing. CUDA. Agent-based modeling. Computational epidemiology. COVID-19.",
      indent=False)
doc.add_page_break()

# ============================================================ SUMARIO
centro("SUMÁRIO", bold=True); doc.add_paragraph()
toc()
doc.add_page_break()

# ============================================================ 1 INTRODUÇÃO
h1("1 INTRODUÇÃO")
corpo("A pandemia de COVID-19 evidenciou a importância de modelos computacionais capazes de antecipar a "
      "dinâmica de epidemias e de avaliar o impacto de políticas de contenção. Entre as abordagens "
      "disponíveis, os modelos baseados em agentes (ABM) destacam-se por representar explicitamente "
      "indivíduos, suas interações e a estrutura espacial da população, capturando heterogeneidades que os "
      "modelos compartimentais clássicos, baseados em equações diferenciais, descrevem apenas de forma "
      "agregada.")
corpo("Essa riqueza de detalhe tem um preço: o custo computacional de um ABM cresce com o número de "
      "agentes e de passos de tempo, e estudos estatisticamente robustos exigem dezenas de réplicas "
      "independentes. Em cidades grandes, com milhões de habitantes, uma única execução pode levar horas em "
      "um processador convencional, inviabilizando a exploração de parâmetros e a análise de incerteza. "
      "Nesse contexto, a computação paralela em unidades de processamento gráfico (GPUs), por meio da "
      "plataforma CUDA, surge como alternativa natural: o caráter local e massivamente paralelo das "
      "atualizações de cada agente é, em princípio, adequado à arquitetura de milhares de núcleos de uma "
      "GPU.")
corpo("A paralelização, contudo, só é útil se preservar a fidelidade do modelo: os resultados da versão "
      "em GPU devem convergir para os da implementação serial de referência. Garantir essa equivalência, "
      "diante de diferenças inevitáveis entre as duas arquiteturas — geração de números aleatórios "
      "por thread, ordem de execução, condições de corrida — é o desafio central deste trabalho.")
h2("1.1 Objetivos")
corpo("O objetivo geral é paralelizar em CUDA uma simulação epidemiológica baseada em agentes de "
      "COVID-19 em áreas urbanas brasileiras e validar que os resultados da GPU reproduzem os da "
      "implementação serial de referência, medindo o ganho de desempenho obtido.")
corpo("Como objetivos específicos, têm-se: (i) implementar na GPU os dois mecanismos de contágio e a "
      "máquina de estados de saúde do modelo; (ii) diagnosticar e corrigir as causas de qualquer "
      "divergência entre as duas implementações; (iii) validar quantitativamente a equivalência das curvas "
      "epidêmicas nas quatro cidades; (iv) medir o speedup e a eficiência da GPU; e (v) caracterizar o "
      "gargalo de desempenho por meio de perfilamento e da comparação entre placas distintas.")
h2("1.2 Organização do trabalho")
corpo("O Capítulo 2 apresenta a fundamentação teórica. O Capítulo 3 descreve o modelo, as duas "
      "implementações e os métodos de validação e de benchmark. O Capítulo 4 reúne e discute os "
      "resultados. O Capítulo 5 traz as conclusões e os trabalhos futuros.")

# ============================================================ 2 FUNDAMENTAÇÃO
h1("2 FUNDAMENTAÇÃO TEÓRICA")
h2("2.1 Modelos epidemiológicos compartimentais")
corpo("Os modelos compartimentais dividem a população em classes segundo o estado em relação à doença. O "
      "modelo SIR de Kermack e McKendrick (1927) separa Suscetíveis, Infectados e Recuperados; o modelo "
      "SEIR acrescenta a classe de Expostos (infectados ainda não infecciosos). Um parâmetro central é o "
      "número básico de reprodução R₀, que mede quantas infecções secundárias um caso gera, em média, "
      "numa população totalmente suscetível; quando R₀ > 1 a epidemia cresce (HETHCOTE, 2000). Este "
      "trabalho adota uma variante estendida do SEIR, com estados pré-sintomático, assintomático e graus de "
      "severidade, além de hospitalização e UTI.")
h2("2.2 Modelos baseados em agentes e autômatos celulares")
corpo("Modelos baseados em agentes representam cada indivíduo como uma entidade com estado próprio que "
      "interage com vizinhos e contatos aleatórios. Quando os agentes ocupam células de um reticulado "
      "regular e evoluem por regras locais, o modelo aproxima-se de um autômato celular (WOLFRAM, 1984). A "
      "estrutura espacial introduz correlações que afetam a dinâmica — por exemplo, suavizando a "
      "transição abrupta entre extinção e surto observada em modelos de campo médio.")
h2("2.3 Computação paralela em GPU e CUDA")
corpo("Uma GPU moderna possui milhares de núcleos organizados em multiprocessadores, projetados para "
      "executar a mesma operação sobre muitos dados simultaneamente (modelo SIMT). A plataforma CUDA, da "
      "NVIDIA, expõe esse paralelismo por meio de funções chamadas kernels, executadas por milhares de "
      "threads (NVIDIA, 2024; KIRK; HWU, 2016). O bom desempenho depende de ocupar os núcleos e, "
      "sobretudo, de usar eficientemente a hierarquia de memória — registradores, cache L1/L2 e a "
      "memória global (DRAM), esta última com latência alta e largura de banda limitada.")
h2("2.4 Métricas de desempenho")
corpo("O ganho da paralelização é medido pelo speedup, razão entre o tempo da versão serial e o da "
      "versão paralela para a mesma carga. Aplicações podem ser limitadas por computação (compute-bound) "
      "ou por memória (memory-bound); no segundo caso, o desempenho escala com a largura de banda, e não "
      "com o número de operações em ponto flutuante. O modelo Roofline (WILLIAMS; WATERMAN; PATTERSON, "
      "2009) formaliza essa distinção. A identificação correta do gargalo orienta as otimizações que de "
      "fato produzem ganho.")

# ============================================================ 3 METODOLOGIA
h1("3 MATERIAIS E MÉTODOS")
h2("3.1 O modelo epidemiológico")
corpo("O modelo é um ABM SEIR estendido sobre um reticulado de lado L com condições de contorno "
      "periódicas. Cada célula é um indivíduo com um estado de saúde que evolui na sequência: suscetível "
      "(S), exposto (E), pré-sintomático (IP), assintomático (IA) ou sintomático em graus de severidade "
      "(leve, moderado, grave), podendo seguir para hospital (H), UTI e óbito por COVID, ou recuperar-se "
      "(R). Mortes naturais ocorrem em qualquer estado, repondo a célula por um novo suscetível. As "
      "durações de cada estado e as probabilidades de transição seguem a literatura clínica adotada no "
      "modelo de referência.")
corpo("A transmissão ocorre por dois mecanismos simultâneos: (i) o suscetível procura infectados entre "
      "seus vizinhos (vizinhança de Moore, 8 células, em alta densidade; de Von Neumann, 4 células, em "
      "baixa densidade) e entre contatos aleatórios na malha; e (ii) o infectado procura suscetíveis entre "
      "contatos aleatórios. A probabilidade de contágio é P = 1 − (1 − β)ⁿ, em que n é o "
      "número de contatos infecciosos e β é o parâmetro de transmissão da doença. As quatro cidades "
      "diferem em tamanho, densidade, número de contatos aleatórios e capacidade hospitalar (Tabela 1).")
tabela("Parâmetros das quatro cidades simuladas",
       ["Cidade", "L", "Densidade", "Vizinhança", "Contatos aleat."],
       [["São Paulo", "3355", "Alta", "Moore (8)", "2–19"],
        ["Rocinha", "264", "Alta", "Moore (8)", "2–120"],
        ["Brasília", "1604", "Baixa", "Von Neumann (4)", "2"],
        ["Manaus", "1343", "Baixa", "Von Neumann (4)", "2"]],
       fonte="Adaptado do modelo de referência (2026).")
h2("3.2 Implementação serial de referência")
corpo("A implementação serial, em C, percorre o reticulado célula a célula a cada dia, despachando a "
      "função correspondente ao estado de saúde e, ao final do dia, atualizando o reticulado por uma "
      "função de update. É a referência contra a qual a versão paralela é validada.")
h2("3.3 Paralelização em CUDA")
corpo("Na versão em GPU, cada estado é tratado por um kernel próprio (S, E, IP, IS, H, ICU), executado "
      "por uma thread por célula, seguido de um kernel de atualização. Emprega-se duplo buffer (lê-se o "
      "campo Health e escreve-se em um campo Swap, copiado no update) para garantir um instantâneo "
      "consistente do dia. Cada thread mantém seu próprio gerador de números aleatórios. Para evitar dupla "
      "infecção no mesmo passo, usam-se marcadores atômicos. Uma otimização de layout de memória "
      "(Structure-of-Arrays) mantém o campo de saúde em um vetor compacto de um byte por célula, que cabe "
      "no cache L2, acelerando os acessos aleatórios.")
h2("3.4 Calibração do parâmetro de transmissão")
corpo("Para cada cidade, calibrou-se β de modo que o número básico de reprodução fosse R₀ "
      "≈ 3,5, valor de referência para a COVID-19. A calibração foi feita medindo-se o número médio "
      "de infecções secundárias geradas por um caso-índice em população totalmente suscetível.")
h2("3.5 Ambiente computacional")
corpo("As execuções principais usaram uma GPU NVIDIA GeForce RTX 4070 SUPER (arquitetura Ada Lovelace, "
      "7168 núcleos CUDA, 48 MB de cache L2, 504 GB/s de largura de banda, 12 GB de memória), com "
      "compilação pelo nvcc (CUDA 12.6) e otimização ‑O3/‑O2. A versão serial foi compilada com "
      "as mesmas otimizações de host. A comparação entre placas empregou ainda uma GeForce GTX 1050 Ti "
      "(Pascal, 768 núcleos, 112 GB/s, 4 GB).")
h2("3.6 Método de validação e de benchmark")
corpo("A validação compara as curvas de prevalência da versão serial e da GPU sob o mesmo β, com "
      "L real de cada cidade e 50 simulações independentes, quantificando a diferença pelo erro quadrático "
      "médio (RMSE) da fração de suscetíveis ao longo dos 400 dias. O benchmark mede o tempo de parede de "
      "cada implementação para a mesma epidemia; o perfilamento de tempo por kernel/função usa "
      "temporizadores de alta resolução. A variabilidade é avaliada pela distribuição da taxa de ataque "
      "entre as 50 réplicas.")

# ============================================================ 4 RESULTADOS
h1("4 RESULTADOS E DISCUSSÃO")
h2("4.1 Correções de infraestrutura na GPU")
corpo("Antes da validação epidemiológica, três problemas de infraestrutura precisaram ser resolvidos. "
      "Primeiro, o mecanismo de Detecção e Recuperação de Timeout (TDR) do Windows reiniciava o driver "
      "gráfico sempre que um kernel ultrapassava dois segundos, fazendo as execuções maiores falharem; a "
      "solução foi elevar o limite do TDR e adicionar verificação de erro CUDA. Segundo, o gerador de "
      "números aleatórios original, um congruente puramente multiplicativo, degenerava em ciclos curtos "
      "para certas sementes, levando laços de rejeição a laços infinitos; adicionaram-se limites de "
      "segurança. Terceiro, a cidade de São Paulo, com vetor de 721 MB muito maior que o cache, era "
      "dominada por acessos aleatórios à memória; a otimização Structure-of-Arrays do campo de saúde "
      "reduziu o tempo por simulação em ordem de grandeza.")
h2("4.2 Diagnóstico e correção da divergência serial–GPU")
corpo("Sob o mesmo β, as duas implementações divergiam: a GPU produzia epidemias mais intensas. Uma "
      "bissecção empírica isolou a origem desligando cada mecanismo de cada vez e medindo o resultado, "
      "descartando por medição as hipóteses de gerador aleatório, de duração infecciosa e de parâmetros. A "
      "investigação revelou que a divergência vinha de dois defeitos na implementação serial — a GPU "
      "estava correta. O primeiro, responsável por cerca de 92% da diferença, era um isolamento social de "
      "50% da população ativado por engano: a diretiva de pré-processador #if(BeginOfIsolation==ON) era "
      "sempre verdadeira, pois o pré-processador avaliava símbolos não definidos como zero. O segundo era "
      "uma sobrescrita na função do suscetível que apagava infecções já causadas pelo mecanismo "
      "infectado-dirigido. Corrigidos os dois pontos no código serial, as implementações passaram a "
      "coincidir sob o mesmo β.")
h2("4.3 Validação da equivalência")
corpo("Com as correções, serial e GPU produzem a mesma epidemia sob o mesmo β nas quatro cidades, no "
      "tamanho real e com 50 simulações. A Figura 1 sobrepõe as curvas de prevalência (linha cheia para a "
      "GPU, tracejada para o serial): elas praticamente coincidem. A Tabela 2 quantifica a diferença pelo "
      "erro quadrático médio da fração de suscetíveis, inferior a 1,2 ponto percentual em todas as cidades "
      "(e abaixo de 0,03 ponto percentual para os infecciosos), compatível com ruído estatístico. A "
      "Figura 2 resume esse erro residual por cidade.")
figura("Curvas de prevalência — comparação Serial (tracejado) × GPU (linha cheia) por cidade",
       os.path.join(G, "resultados", "comparacao", "prevalencia.png"), largura=16)
tabela("Validação da equivalência sob o mesmo β (MAXSIM = 50)",
       ["Cidade", "β", "Ataque GPU/Serial", "RMSE de S(t)"],
       [["Rocinha", "0,0049", "58,7% / 58,8%", "0,76 p.p."],
        ["Manaus", "0,0995", "75,9% / 75,4%", "0,86 p.p."],
        ["Brasília", "0,0995", "76,8% / 76,3%", "1,13 p.p."],
        ["São Paulo", "0,0243", "72,6% / 72,8%", "0,74 p.p."]])
figura("Erro residual (RMSE da fração de suscetíveis) entre serial e GPU, por cidade",
       os.path.join(G, "validacao", "equivalencia_rmse.png"), largura=12)
corpo("Esse resultado tem uma implicação metodológica importante: não há necessidade de calibrar um "
      "β diferente para cada implementação — a calibração separada que se mostrava necessária "
      "antes apenas compensava os dois defeitos do serial.")
h2("4.4 Resultados epidemiológicos")
corpo("A Figura 3 apresenta as curvas dos compartimentos principais nas quatro cidades. A taxa de ataque "
      "final varia de cerca de 59% (Rocinha) a 77% (Brasília), refletindo as diferenças de densidade, "
      "vizinhança e número de contatos. Os picos de infecciosos ocorrem entre os dias 170 e 230, com "
      "epidemias de propagação relativamente lenta no reticulado.")
figura("Curvas de prevalência por cidade (compartimentos e carga clínica)",
       os.path.join(G, "resultados", "serial", "prevalencia.png"), largura=16)
h2("4.5 Desempenho: speedup")
corpo("Medido sobre a mesma epidemia (mesmo β) e 50 simulações, o speedup da GPU sobre a CPU serial "
      "cresce com o tamanho do reticulado, de 1,8 vez na Rocinha a 56 vezes em São Paulo (Tabela 3, "
      "Figura 4). Em grades pequenas a GPU é subutilizada e dominada pelo custo fixo de lançamento de "
      "kernels; o ganho só se materializa quando há paralelismo suficiente para ocupar seus núcleos.")
tabela("Tempo de execução e speedup por cidade (MAXSIM = 50)",
       ["Cidade", "Células", "Serial", "GPU", "Speedup"],
       [["Rocinha", "70 mil", "235 s", "129 s", "1,8×"],
        ["Manaus", "1,80 mi", "1.891 s", "136 s", "13,9×"],
        ["Brasília", "2,57 mi", "2.124 s", "131 s", "16,2×"],
        ["São Paulo", "11,25 mi", "35.755 s", "634 s", "56,4×"]])
figura("Speedup da GPU sobre a CPU serial, por cidade",
       os.path.join(G, "speedup", "speedup_por_cidade.png"), largura=14)
h2("4.6 Análise de hardware: a simulação é limitada por memória")
corpo("Três evidências independentes mostram que o gargalo é a memória, não o cálculo. O perfilamento "
      "(Figura 5) revela que o kernel do suscetível — que faz os acessos aleatórios à memória para "
      "verificar contatos — domina o tempo (60% na GPU, 88% no serial). O tamanho do bloco de threads "
      "(de 64 a 1024) praticamente não altera o tempo, indicando que a ocupância não é o limitante. E a "
      "pegada de memória (Figura 6) explica a otimização Structure-of-Arrays: o vetor completo de agentes "
      "ultrapassa o cache L2 já em L ≈ 866, enquanto o vetor compacto de saúde cabe até L ≈ 6900.")
figura("Perfil de execução: distribuição do tempo por kernel (GPU) e por função (serial)",
       os.path.join(G, "hardware", "profile_serial_gpu.png"), largura=15)
figura("Pegada de memória dos vetores frente aos limites da GPU (cache L2 e VRAM)",
       os.path.join(G, "hardware", "memoria_vs_L.png"), largura=13)
h2("4.7 Comparação entre GPUs")
corpo("Executar a mesma simulação em uma segunda placa (GTX 1050 Ti) confirma duas conclusões "
      "(Figura 7). Primeiro, a taxa de ataque é idêntica nas duas GPUs — o resultado independe do "
      "hardware, reforçando a correção da implementação. Segundo, a 1050 Ti é cerca de 4 a 6 vezes mais "
      "lenta nas cidades grandes, razão que coincide com a de largura de banda de memória (4,5 vezes) e "
      "não com a de poder de cálculo em ponto flutuante (cerca de 17 vezes) — confirmando, por outra "
      "via, que a simulação é limitada por memória.")
figura("Comparação entre GPUs (RTX 4070 SUPER × GTX 1050 Ti): tempo e razão frente aos limites teóricos",
       os.path.join(G, "hardware", "comparacao_gpus.png"), largura=16)
h2("4.8 Variabilidade estatística")
corpo("A análise das 50 réplicas por cidade mostra que, no regime calibrado (R₀ = 3,5), os "
      "resultados são altamente robustos: o desvio-padrão da taxa de ataque é de no máximo 0,9 ponto "
      "percentual, e todas as réplicas produzem surtos — não há bimodalidade. A maior variabilidade "
      "está no instante do pico, não no desfecho. Próximo ao limiar epidêmico, porém, a incerteza cresce "
      "cerca de sete vezes, com extinções estocásticas (Figura 8). Conclui-se que a calibração para "
      "R₀ = 3,5 coloca a simulação num regime reprodutível, afastado da zona de alta incerteza.")
figura("Distribuição da taxa de ataque na Rocinha: incerteza alta próximo ao limiar × robustez no regime calibrado",
       os.path.join(G, "variabilidade", "robustez_vs_limiar.png"), largura=16)

# ============================================================ 5 CONCLUSÃO
h1("5 CONCLUSÃO")
corpo("Este trabalho paralelizou em CUDA uma simulação epidemiológica baseada em agentes de COVID-19 e "
      "validou que a implementação em GPU reproduz fielmente a versão serial. A principal contribuição "
      "metodológica foi mostrar que a divergência inicial entre as duas implementações não era intrínseca "
      "à paralelização, mas decorria de dois defeitos na implementação serial de referência, identificados "
      "por uma investigação empírica sistemática. Corrigidos esses defeitos, as duas versões produzem a "
      "mesma epidemia sob o mesmo parâmetro de transmissão, com diferença compatível com ruído "
      "estatístico, dispensando calibrações separadas.")
corpo("Do ponto de vista de desempenho, a GPU proporcionou ganhos de até 56 vezes na maior cidade, com o "
      "speedup crescendo com o tamanho do problema. A caracterização do gargalo — por perfilamento, "
      "invariância ao tamanho de bloco e comparação entre duas GPUs — demonstrou que a simulação é "
      "limitada pela largura de banda de memória, o que justificou e quantificou o ganho de uma otimização "
      "de layout de dados. A análise estatística confirmou a robustez dos resultados no regime calibrado.")
corpo("Como trabalhos futuros, destacam-se: a avaliação de cenários de intervenção não farmacêutica "
      "(isolamento social) agora que o mecanismo foi corrigido; a substituição do gerador de números "
      "aleatórios por um de período completo, com recalibração; a extensão a outras cidades e a grades "
      "maiores; e a exploração de técnicas adicionais de otimização de memória e de fusão de kernels.")

# ============================================================ REFERÊNCIAS
doc.add_page_break()
centro("REFERÊNCIAS", bold=True); doc.add_paragraph()
refs = [
 "FERGUSON, N. M. et al. Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand. London: Imperial College COVID-19 Response Team, 2020.",
 "HETHCOTE, H. W. The mathematics of infectious diseases. SIAM Review, v. 42, n. 4, p. 599–653, 2000.",
 "KERMACK, W. O.; McKENDRICK, A. G. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London A, v. 115, n. 772, p. 700–721, 1927.",
 "KIRK, D. B.; HWU, W. W. Programming massively parallel processors: a hands-on approach. 3. ed. Waltham: Morgan Kaufmann, 2016.",
 "NVIDIA CORPORATION. CUDA C++ programming guide. Santa Clara: NVIDIA, 2024.",
 "WILLIAMS, S.; WATERMAN, A.; PATTERSON, D. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, v. 52, n. 4, p. 65–76, 2009.",
 "WOLFRAM, S. Cellular automata as models of complexity. Nature, v. 311, p. 419–424, 1984.",
 "[Referência do modelo epidemiológico de base — a ser completada pelo autor com a publicação do orientador.]",
]
for ref in refs:
    p = doc.add_paragraph(ref); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.space_after = Pt(12)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("OK ->", OUT)
print("paginas/figuras:", _figN[0], "figuras,", _tabN[0], "tabelas")
