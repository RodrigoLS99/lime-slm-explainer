import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from openai import OpenAI

# =====================================================================
# 1. O Modelo "Caixa-Preta" (Treinado com Dados Reais da B2W)
# =====================================================================
print("Baixando CSV de dados reais da B2W (Lojas Americanas/Submarino)...")
url = "https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/main/B2W-Reviews01.csv"
df = pd.read_csv(url, usecols=['review_text', 'recommend_to_a_friend'])

# Limpeza: remove linhas vazias
df = df.dropna()

# Converte recomendação em Sentimento (1 = Positivo, 0 = Negativo)
df['sentimento'] = df['recommend_to_a_friend'].apply(lambda x: 1 if x == 'Yes' else 0)

# Separa 5000 amostras para o Treino e 500 para o Teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    df['review_text'], 
    df['sentimento'], 
    train_size=5000, 
    test_size=500,
    random_state=42
)

textos_treino = X_treino.tolist()
y_treino_lista = y_treino.tolist()

print("Treinando o modelo de Machine Learning (apenas nos dados de treino)...")
vec = TfidfVectorizer(max_features=5000)
clf = LogisticRegression(random_state=42, max_iter=500)
meu_modelo = make_pipeline(vec, clf)
meu_modelo.fit(textos_treino, y_treino_lista)
print("Modelo treinado com sucesso!\n")


# =====================================================================
# 2. Configuração do SLM Local (Docker Model Runner)
# =====================================================================
cliente_slm = OpenAI(
    base_url="http://127.0.0.1:12434/v1", 
    api_key="local-docker" 
)

NOME_SLM = 'ai/qwen2.5' 

def gerar_explicacao_slm(frase, pesos_lime):
    """
    Usa o SLM para traduzir a saída matemática do LIME em texto natural.
    """
    pesos_str = "\n".join([f"- Palavra: '{palavra}' | Peso: {peso:.4f}" for palavra, peso in pesos_lime])
    
    prompt = f"""
Você é um especialista em interpretabilidade de Inteligência Artificial.
O nosso modelo analisou a frase abaixo e classificou o sentimento dela. O algoritmo LIME atribuiu pesos matemáticos para as palavras mais importantes (pesos positivos puxam para a classe Positiva, negativos para a Negativa).

Frase do usuário: "{frase}"

Pesos calculados pelo LIME:
{pesos_str}

Com base EXCLUSIVAMENTE nesses pesos, escreva um parágrafo curto (máximo 3 frases) explicando de forma clara e simples para um usuário leigo por que o modelo tomou essa decisão.
"""

    try:
        resposta = cliente_slm.chat.completions.create(
            model=NOME_SLM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1 
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro ao conectar com o SLM ({NOME_SLM}): {e}"


# =====================================================================
# 3. Varredura Adaptativa LIME + Geração de Texto
# =====================================================================
def calcular_jaccard(lista1, lista2):
    set1, set2 = set(lista1), set(lista2)
    if not set1.union(set2): return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def avaliar_explicabilidade_adaptativa(text_query, model_predict_fn, n_steps=[10, 50, 100, 200, 500, 1000]):
    explainer = LimeTextExplainer(class_names=['Negativo', 'Positivo'])
    
    resultados = []
    top_features_anterior = []
    
    for n in n_steps:
        exp = explainer.explain_instance(text_query, model_predict_fn, num_samples=n)
        
        # Pega as 4 palavras mais importantes
        pesos_completos = exp.as_list()
        top_features_atual = [palavra for palavra, peso in pesos_completos[:4]]
        
        # Calcula estabilidade matemática (Jaccard)
        jaccard = calcular_jaccard(top_features_anterior, top_features_atual)
        
        # Gera o texto com o SLM baseado na saída atual do LIME
        explicacao_texto = gerar_explicacao_slm(text_query, pesos_completos[:4])
        
        resultados.append({
            'N_Perturbacoes': n,
            'Estabilidade_Jaccard': jaccard,
            'Top_Palavras': top_features_atual,
            'Explicacao_SLM': explicacao_texto
        })
        
        top_features_anterior = top_features_atual
        
    return resultados


# =====================================================================
# 4. Execução do Experimento (Com Dados Reais de Teste)
# =====================================================================
if __name__ == "__main__":
    
    # Sorteia 10 frases aleatórias do conjunto de TESTE (nunca vistas pelo modelo)
    frases_teste = X_teste.sample(100, random_state=42).tolist()
    
    todos_resultados = []
    
    print("\n================ INICIANDO BATERIA DE TESTES ================\n")

    for i, frase in enumerate(frases_teste, 1):
        print(f"[{i}/{len(frases_teste)}] Processando frase de teste: '{frase[:50]}...'")
        
        # Executa o estudo variando N
        dados_frase = avaliar_explicabilidade_adaptativa(frase, meu_modelo.predict_proba)
        
        # Adiciona os dados da frase na lista geral
        for linha in dados_frase:
            linha['ID_Frase'] = i
            linha['Frase_Original'] = frase
            todos_resultados.append(linha)
            
    # Salva os resultados
    df_resultados = pd.DataFrame(todos_resultados)
    
    # Reorganiza as colunas para o CSV ficar mais legível
    colunas_ordem = ['ID_Frase', 'Frase_Original', 'N_Perturbacoes', 'Estabilidade_Jaccard', 'Top_Palavras', 'Explicacao_SLM']
    df_resultados = df_resultados[colunas_ordem]
    
    df_resultados.to_csv("avaliacao_adaptativa_teste_b2w.csv", index=False)
    
    print("\n================ EXPERIMENTO CONCLUÍDO ================")
    print("Dados de todas as frases exportados para 'avaliacao_adaptativa_teste_b2w.csv' com sucesso!")