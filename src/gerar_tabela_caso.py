import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# 1. Carregar os dados
try:
    df = pd.read_csv('avaliacao_adaptativa_teste_b2w.csv')
except FileNotFoundError:
    print("Arquivo não encontrado. Certifique-se de que o nome do CSV está correto.")
    exit()

# 2. Selecionar a primeira frase como Estudo de Caso
id_caso = df['ID_Frase'].iloc[0]
df_caso = df[df['ID_Frase'] == id_caso].copy()
frase_original = df_caso.iloc[0]['Frase_Original']

# 3. Preparar os dados da tabela
colunas = ['N_Perturbacoes', 'Estabilidade_Jaccard', 'Top_Palavras', 'Explicacao_SLM']
df_tabela = df_caso[colunas].copy()

# Formatar o texto para caber bem na tabela (textwrap quebra linhas longas)
df_tabela['Explicacao_SLM'] = df_tabela['Explicacao_SLM'].apply(lambda x: textwrap.fill(str(x), width=65))
df_tabela['Top_Palavras'] = df_tabela['Top_Palavras'].apply(lambda x: textwrap.fill(str(x), width=35))
df_tabela['Estabilidade_Jaccard'] = df_tabela['Estabilidade_Jaccard'].apply(lambda x: f"{x:.2f}")

# Renomear as colunas para a apresentação visual
df_tabela.columns = ['N', 'Jaccard', 'Top Palavras (LIME)', 'Explicação Gerada (Qwen2.5)']

# =====================================================================
# SAÍDA 1: Terminal (Markdown)
# =====================================================================
print("\n" + "="*90)
print(" ESTUDO DE CASO: EVOLUÇÃO DA EXPLICABILIDADE ADAPTATIVA")
print(f" Frase: '{frase_original}'")
print("="*90 + "\n")
print(df_tabela.to_markdown(index=False))
print("\n" + "="*90 + "\n")

# =====================================================================
# SAÍDA 2: Exportar como Imagem (Para o Artigo)
# =====================================================================
# Ajustar o tamanho da figura dependendo de quantas linhas temos
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Criar a tabela no Matplotlib
tabela = ax.table(
    cellText=df_tabela.values,
    colLabels=df_tabela.columns,
    cellLoc='left',
    loc='center'
)

tabela.auto_set_font_size(False)
tabela.set_fontsize(11)
tabela.scale(1, 4.5)  # Aumenta a altura das linhas para acomodar os parágrafos

# Estilizar a tabela
for (i, j), cell in tabela.get_celld().items():
    # Cabeçalho
    if i == 0:
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#4c72b0')
    # Células de dados
    else:
        # Pega o valor do Jaccard da linha atual
        jaccard_val = float(df_tabela.iloc[i-1]['Jaccard'])
        
        # Destaca a linha inteira com um azul claro se Jaccard >= 0.9 (Estabilidade)
        if jaccard_val >= 0.90:
            cell.set_facecolor('#e6f2ff')
            if j == 1: # Destaca o valor do Jaccard em negrito
                cell.set_text_props(weight='bold', color='#004488')
        
        # Centralizar colunas 'N' e 'Jaccard'
        if j in [0, 1]:
            cell.set_text_props(ha='center')

plt.title(f"Evolução da Explicabilidade Textual\nFrase: '{frase_original}'", 
          fontweight="bold", fontsize=14, pad=20)

# Salvar
plt.tight_layout()
plt.savefig('tabela_evolucao_texto.png', dpi=300, bbox_inches='tight')
print("-> Imagem da tabela salva com sucesso como 'tabela_evolucao_texto.png'")