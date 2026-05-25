# Decisions — Pipeline PLN Olist
### Decisões arquiteturais e técnicas do projeto

---

## D01 — Dataset: Olist Brazilian E-Commerce

**Decisão**: Usar o dataset Olist (Kaggle) como corpus principal.

**Alternativas consideradas**: B2W Reviews (PT-BR), BBC News (EN), AG News (EN), Consumer Complaints CFPB (EN).

**Justificativa**:
- Corpus em português demonstra competência técnica adicional (RSLPStemmer, `pt_core_news_sm`, stopwords PT)
- Rótulos naturais (1-5 estrelas) → classificação supervisionada sem rotulação manual
- Múltiplas tabelas integrávels criam riqueza de metadados (categoria, estado, preço) para o grafo
- Domínio rico para NER: cidades, estados, valores monetários, marcas de transporte
- Tripla `categoria → tópico → região` permite grafo com semântica de negócio clara
- Dataset público e citável no Kaggle com API de download

**Tradeoff aceito**: Textos curtos (~50-80 palavras) limitam topic modeling. Mitigação: concatenar title + message, filtrar reviews < 20 tokens, agregar por categoria para topic modeling.

---

## D02 — Estratégia de Labels: Binária como eixo principal

**Decisão**: Usar classificação binária {4,5}→positive, {1,2}→negative como eixo principal. Excluir neutros (3 estrelas).

**Alternativas consideradas**: Ternária (positivo/neutro/negativo), multiclasse (5 classes).

**Justificativa**:
- Classificação binária produz F1 mais alto e mais fácil de interpretar para o professor
- A classe neutra (score 3) é ambígua e polui a fronteira de decisão
- Reduz o impacto do desbalanceamento (maioria 5 estrelas + minoria 1 estrela)
- Permite comparação limpa entre modelos no relatório

**Tradeoff aceito**: Perda de granularidade. Mitigação: topic modeling usa corpus completo (sem excluir neutros).

---

## D03 — Pré-processamento: Lemmatização como estratégia principal

**Decisão**: Usar lemmatização (spaCy `pt_core_news_sm`) como estratégia padrão para classificação e topic modeling. Stemming (RSLPStemmer) implementado apenas para comparação.

**Justificativa**:
- Lemas são interpretáveis ("entregaram" → "entregar" vs "entreg" do stemmer)
- Preserva morfologia suficiente para semântica (adjetivos mantêm gênero/número relevante)
- spaCy usa modelo morfológico treinado, não heurísticas simples
- Topic modeling com lemas gera tópicos mais legíveis

**Tradeoff aceito**: Lemmatização é mais lenta que stemming. Mitigação: processar em batch com `nlp.pipe(batch_size=256)` e salvar `df_processed.pkl`.

---

## D04 — Vetorização: TF-IDF com bigramas como feature principal

**Decisão**: Usar `TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True)` como representação primária para classificação e busca.

**Justificativa**:
- Bigramas capturam negações críticas em português: "não chegou", "muito ruim", "péssimo atendimento"
- `sublinear_tf=True` usa log(tf) → reduz dominância de termos muito frequentes
- TF-IDF supera BoW para textos curtos porque penaliza termos ubíquos
- `max_features=15000` equilibra expressividade e eficiência de memória
- BoW implementado para LDA (que requer contagens brutas)

**Tradeoff aceito**: Bigramas aumentam dimensionalidade (15k > 10k). A esparsidade aumenta mas o ganho de expressividade justifica.

---

## D05 — Word2Vec: Skip-gram com 100 dimensões

**Decisão**: `Word2Vec(sg=1, vector_size=100, window=5, min_count=5, epochs=10)`.

**Justificativa**:
- Skip-gram (sg=1) performa melhor em corpus menores e captura relações raras melhor que CBOW
- 100 dimensões: suficiente para 41k documentos sem overfitting; menor = mais rápido
- `min_count=5` evita vetores de palavras raras com pouco suporte estatístico
- `window=5` captura contexto local de e-commerce (negações próximas ao alvo)

**Tradeoff aceito**: Não usamos embeddings pré-treinados (BERT, BERTimbau). Justificativa: demonstra construção do zero, mais coerente com a disciplina. BERTimbau seria overkill e lento sem GPU.

---

## D06 — Classificadores: NB (baseline) → LR → SVM (principal)

**Decisão**: Implementar Naive Bayes multinomial como baseline, Logistic Regression para interpretabilidade, e LinearSVC como modelo principal.

**Justificativa de cada modelo**:

| Modelo | Papel | Por que |
|--------|-------|---------|
| `MultinomialNB` | Baseline | Rápido, explica o piso de desempenho; viola independência mas é referência clássica |
| `LogisticRegression` | Interpretável | Coeficientes = importância de features; detecta quais palavras são mais discriminativas |
| `LinearSVC` | Principal | Ótimo para TF-IDF esparso de alta dimensão; baseado em margem máxima; mais robusto a outliers |

**Expectativa de F1 macro**: NB ~0.88-0.92 → LR ~0.92-0.95 → SVM ~0.93-0.96.

**Tradeoff aceito**: Sem Random Forest nem XGBoost — mais lentos e sem ganho justificado em TF-IDF esparso. Sem cross-validation explícita (custo computacional; rubrica não exige).

---

## D07 — Topic Modeling: LDA (BoW) + NMF (TF-IDF), 8 tópicos

**Decisão**: Usar LDA sobre BoW e NMF sobre TF-IDF, ambos com 8 componentes.

**Justificativa**:
- LDA: usa BoW porque a formulação probabilística assume contagens (não pesos TF-IDF)
- NMF: usa TF-IDF porque a fatoração matricial funciona melhor com dados normalizados
- 8 tópicos: escolhido empiricamente alinhado com os 8 padrões temáticos esperados do corpus Olist (ver SPEC §8.3)
- Comparar os dois: NMF tende a tópicos mais coesos; LDA é mais probabilístico — evidência da diferença é academicamente relevante

**Tradeoff aceito**: Não implementamos seleção de número de tópicos por coherence score (custo computacional). Mencionamos como limitação.

---

## D08 — Grafo de Conhecimento: NetworkX + PyVis

**Decisão**: Construir grafo com NetworkX (análise), exportar como HTML interativo com PyVis.

**Tipos de nós**: CATEGORY (15 top), TOPIC (8 tópicos LDA), STATE (10 top estados), SENTIMENT (3 nós fixos). Total: ≥36 nós.

**Justificativa**:
- NetworkX: padrão para análise de grafos em Python; permite cálculo de centralidade, pagerank, betweenness
- PyVis: visualização interativa no navegador sem JavaScript manual; impressiona visualmente
- 36 nós >> exigência de 20 da rubrica — folga confortável

**Tradeoff aceito**: Grafo não-direcionado por simplicidade. Pesos derivados de estatísticas do corpus (% de reviews, volume), não de embeddings.

---

## D09 — Estratégia de Reprodutibilidade

**Decisão**: `random_state=42` global, paths relativos, `df_main.pkl` e `df_processed.pkl` como checkpoints, amostra pública de 1000 linhas.

**Justificativa**:
- `pkl` checkpoints evitam re-processamento em cada execução (spaCy sobre 41k docs leva minutos)
- `data/sample/` permite avaliadores sem acesso ao Kaggle verificarem a estrutura básica
- Imports todos na primeira célula, `data/raw/` no `.gitignore`

---

## D10 — Biblioteca de NER: spaCy `pt_core_news_sm`

**Decisão**: Usar o modelo `pt_core_news_sm` do spaCy para NER e POS tagging.

**Alternativas consideradas**: `pt_core_news_lg` (mais acurado), NLTK NER (não suporta PT-BR nativamente), Hugging Face NER (overkill).

**Justificativa**:
- `sm` é suficiente para demonstração; `lg` exige mais memória sem ganho pedagógico claro
- spaCy fornece displaCy para visualização de entidades inline no Jupyter — requisito visual da rubrica
- Modelo `sm` é mais rápido, adequado para corpus de 41k documentos sem GPU

**Tradeoff aceito**: Menor acurácia de NER que `lg` ou modelos BERT-based. Documentado como limitação.

---

## D11 — Normalização de Entidades: Levenshtein com threshold=3

**Decisão**: Usar `python-Levenshtein` com distância máxima de 3 para normalizar variações de cidades/estados.

**Justificativa**:
- Threshold 3 captura erros tipográficos ("Sao Paulo" vs "São Paulo") sem ser muito permissivo
- Lista canônica de 27 estados + principais capitais como âncoras de normalização
- Simples de implementar e de demonstrar antes/depois

---

## D12 — Paleta de Cores e Estilo Visual

**Decisão**: Paleta fixada no início do notebook como constantes globais `COLORS` + `plt.style.use('seaborn-v0_8-whitegrid')`.

| Sentimento | Cor | Hex |
|------------|-----|-----|
| positive | Verde | `#27ae60` |
| negative | Vermelho | `#e74c3c` |
| neutral | Cinza | `#95a5a6` |
| primary | Azul escuro | `#2c3e50` |

**Justificativa**: Consistência visual entre 21 gráficos sinaliza profissionalismo. Verde/vermelho para sentimento é intuitivo. Seaborn whitegrid é limpo para publicação.
