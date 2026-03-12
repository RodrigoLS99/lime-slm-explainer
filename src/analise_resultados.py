import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de estilo dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 24})
sns.set_theme(style="whitegrid", font_scale=2)

# 1. Carrega os dados gerados pelo seu experimento
try:
    df = pd.read_csv('avaliacao_adaptativa_teste_b2w.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'avaliacao_adaptativa_teste_b2w.csv' não encontrado.")
    exit()

LIMIAR_ESTABILIDADE = 0.6

print("\n" + "="*50)
print(" ANÁLISE DOS RESULTADOS DO EXPERIMENTO")
print("="*50 + "\n")

# =====================================================================
# TABELA 1: Ponto de Estabilização e Economia Computacional
# =====================================================================
# Descobre em qual 'N' cada frase estabilizou pela primeira vez
n_parada_por_frase = []

for id_frase in df['ID_Frase'].unique():
    df_frase = df[df['ID_Frase'] == id_frase].sort_values('N_Perturbacoes')
    
    # Encontra o primeiro N onde Jaccard >= 0.9
    estaveis = df_frase[df_frase['Estabilidade_Jaccard'] >= LIMIAR_ESTABILIDADE]
    
    if not estaveis.empty:
        n_ideal = estaveis.iloc[0]['N_Perturbacoes']
    else:
        n_ideal = df_frase['N_Perturbacoes'].max() # Se não estabilizou, parou no limite máximo
        
    n_parada_por_frase.append({'ID_Frase': id_frase, 'N_Ideal': n_ideal})

df_parada = pd.DataFrame(n_parada_por_frase)

# Cálculos de Economia
n_maximo_testado = df['N_Perturbacoes'].max()
media_n_ideal = df_parada['N_Ideal'].mean()
economia_media = ((n_maximo_testado - media_n_ideal) / n_maximo_testado) * 100

print(f"-> Média de perturbações necessárias para estabilizar: {media_n_ideal:.0f}")
print(f"-> Economia Computacional Média (comparado a fixar N={n_maximo_testado}): {economia_media:.1f}% de processamento poupado!\n")


# =====================================================================
# GRÁFICO 1: Curva Média de Convergência do LIME
# =====================================================================
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df, 
    x='N_Perturbacoes', 
    y='Estabilidade_Jaccard', 
    marker='o', 
    linewidth=2.5,
    errorbar=None,  
    color='#2ca02c' 
)

plt.axhline(y=LIMIAR_ESTABILIDADE, color='red', linestyle='--', label=f'Limiar de Parada ({LIMIAR_ESTABILIDADE})')
plt.xscale('log') # Escala logarítmica para N
plt.xticks(df['N_Perturbacoes'].unique(), df['N_Perturbacoes'].unique())
# plt.title('Estabilidade Média de Jaccard vs. N Perturbações', fontsize=14, pad=15)
plt.xlabel('Número de Perturbações (N) - Escala Log')
plt.ylabel('Similaridade Jaccard Média')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_convergencia_media.png', dpi=300)
print("Gráfico gerado: 'grafico_convergencia_media.png'")


# =====================================================================
# GRÁFICO 2: Distribuição dos Pontos de Parada Adaptativa
# =====================================================================
plt.figure(figsize=(9, 5))
contagem_n = df_parada['N_Ideal'].value_counts().sort_index()

sns.barplot(
    x=contagem_n.index, 
    y=contagem_n.values, 
    palette='Blues_d'
)

# plt.title('Em qual "N" as frases estabilizaram?', fontsize=14, pad=15)
plt.xlabel('Número de Perturbações (N_Ideal)')
plt.ylabel('Quantidade de Frases')

# Adiciona os números em cima das barras
for i, v in enumerate(contagem_n.values):
    plt.text(i, v + 0.1, str(v), ha='center', fontsize=18)

plt.tight_layout()
plt.savefig('grafico_distribuicao_parada.png', dpi=300)
print("Gráfico gerado: 'grafico_distribuicao_parada.png'\n")


# =====================================================================
# TABELA 2: Demonstração de Coerência Textual (Case Study)
# =====================================================================
# Vamos pegar a primeira frase como Estudo de Caso para mostrar a evolução do texto
df_caso = df[df['ID_Frase'] == 25].copy()

print("="*50)
print(" ESTUDO DE CASO: EVOLUÇÃO DO TEXTO DO SLM (Frase 25)")
print(f" Frase Original: {df_caso.iloc[0]['Frase_Original']}")
print("="*50)

for index, row in df_caso.iterrows():
    n = row['N_Perturbacoes']
    jaccard = row['Estabilidade_Jaccard']
    texto = row['Explicacao_SLM'].replace('\n', ' ') # Limpa quebras de linha para o print
    
    # Corta o texto se for muito grande só para caber no terminal
    if len(texto) > 120: texto = texto[:117] + "..."
        
    print(f"[N={n:<4}] Jaccard: {jaccard:.2f} | Texto: {texto}")

print("\nConclusão visual da tabela: Note como o texto do SLM muda bruscamente nos Ns iniciais (Jaccard baixo) e congela nos Ns finais (Jaccard alto).")