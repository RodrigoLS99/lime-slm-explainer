INTERPRETABILIDADE ADAPTATIVA: LIME + SLMs LOCAIS

Este repositório contém o código e os experimentos de um projeto de pesquisa focado em Explainable AI (XAI). O objetivo principal é otimizar a extração de explicações locais do algoritmo LIME para dados textuais e traduzir seus resultados matemáticos para uma linguagem natural acessível.

O projeto introduz uma abordagem adaptativa que utiliza o Índice de Similaridade de Jaccard como critério de parada antecipada (early stopping). Isso permite interromper a geração de perturbações do LIME no momento exato da convergência matemática, poupando recursos computacionais. Para a etapa de tradução semântica, o pipeline integra um Small Language Model (SLM) executado de forma totalmente local e privada.

-------------------------------------------------------------------
ARQUITETURA DO EXPERIMENTO
-------------------------------------------------------------------
1. Modelo Base (Caixa-Preta): Um classificador de sentimentos (Regressão Logística + TF-IDF restrito a 5.000 features) treinado com dados reais de avaliações de e-commerce (Dataset B2W Digital).
2. Explicação Local (LIME): Varredura progressiva do número de perturbações (N = 10, 50, 100, 200, 500, 1000).
3. Métrica de Estabilidade: Cálculo da similaridade de Jaccard entre as Top-4 palavras mais importantes de cada lote para determinar o ponto de estabilização.
4. Tradução Semântica (SLM): O modelo Qwen2.5 recebe os pesos numéricos estabilizados e elabora uma justificativa textual concisa para o usuário final.

-------------------------------------------------------------------
TECNOLOGIAS UTILIZADAS
-------------------------------------------------------------------
* Linguagem e Análise de Dados: Python, Pandas, NumPy.
* Machine Learning: Scikit-Learn, Lime.
* Visualização: Matplotlib, Seaborn.
* LLM/SLM Local: OpenAI Python SDK conectado ao Docker Model Runner.

-------------------------------------------------------------------
ESTRUTURA DO PROJETO
-------------------------------------------------------------------
* main.py: Treina o modelo base, conecta ao SLM, aplica o LIME nas frases de teste e exporta o arquivo avaliacao_adaptativa_teste_b2w.csv.
* src/analise_resultados.py: Lê o arquivo CSV gerado e plota os gráficos de convergência de Jaccard e estabilidade das perturbações.
* avaliacao_adaptativa_teste_b2w.csv: Base de dados resultante contendo as frases, métricas matemáticas e as explicações geradas pelo SLM.

-------------------------------------------------------------------
COMO EXECUTAR
-------------------------------------------------------------------

1. Preparando o SLM Local
O projeto utiliza o Docker Desktop para rodar a arquitetura Qwen2.5 de forma nativa e isolada. No seu terminal, execute:
> docker model pull qwen2.5
> docker model run qwen2.5
(O modelo ficará exposto em http://127.0.0.1:12434/v1)

2. Instalando Dependências
Certifique-se de ter o Python instalado e instale os pacotes necessários:
> pip install pandas numpy scikit-learn lime openai matplotlib seaborn

3. Rodando o Pipeline
Primeiro, execute o script principal para baixar os dados, treinar o modelo e gerar as explicações:
> python main.py

Em seguida, para gerar as visualizações e os gráficos do artigo:
> python src/analise_resultados.py

-------------------------------------------------------------------
REFERÊNCIAS PRINCIPAIS
-------------------------------------------------------------------
* Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
* Arrieta, A. B., et al. (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges.
* Qwen Team (2024). Qwen2.5 Technical Report.