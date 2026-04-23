"""
========================================================
SCRIPT 01 — GERADOR DE DATASET SINTÉTICO
Projeto: Sistema de Predição de Risco Clínico
Disciplina: Machine Learning Aplicado à Saúde
========================================================

Objetivo:
    Gerar um dataset sintético com 2000 registros de pacientes,
    com variáveis biomédicas realistas e uma variável-alvo de risco
    calculada por regras clínicas coerentes.

Bibliotecas utilizadas:
    - numpy  : geração de dados aleatórios com distribuições realistas
    - pandas : organização, manipulação e exportação para CSV
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# 1. CONFIGURAÇÃO DA SEMENTE ALEATÓRIA
# ─────────────────────────────────────────────────────────
# Fixar a semente garante que toda vez que o script rodar,
# os mesmos dados serão gerados (reprodutibilidade).
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 2. DEFINIÇÃO DO TAMANHO DO DATASET
# ─────────────────────────────────────────────────────────
N = 2000  # número total de registros

# ─────────────────────────────────────────────────────────
# 3. GERAÇÃO DOS NOMES FICTÍCIOS
# ─────────────────────────────────────────────────────────
# Lista de nomes simples, sem sobrenomes, para uso fictício.
nomes = [
    "Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Gabriela",
    "Hugo", "Isabela", "João", "Karla", "Lucas", "Mariana", "Nicolas",
    "Olivia", "Paulo", "Renata", "Samuel", "Tatiane", "Ursula",
    "Victor", "Wanda", "Xavier", "Yasmin", "Zara", "Aldo", "Bianca",
    "Caio", "Dara", "Eduardo", "Fernanda", "Gustavo", "Helena", "Igor",
    "Julia", "Kevin", "Laura", "Marcos", "Nathalia", "Omar"
]

# Sorteia N nomes aleatoriamente com reposição (replace=True)
nomes_pacientes = np.random.choice(nomes, size=N, replace=True)

# ─────────────────────────────────────────────────────────
# 4. GERAÇÃO DA VARIÁVEL: IDADE
# ─────────────────────────────────────────────────────────
# Faixa: 18 a 99 anos (adultos até idosos)
# Usamos distribuição uniforme discreta para cobrir toda a faixa.
idade = np.random.randint(18, 100, size=N)  # randint: [low, high)

# ─────────────────────────────────────────────────────────
# 5. GERAÇÃO DA VARIÁVEL: GLICOSE (mg/dL)
# ─────────────────────────────────────────────────────────
# Referência clínica:
#   Normal  : 70 – 99 mg/dL (jejum)
#   Pré-DM  : 100 – 125 mg/dL
#   Diabético: ≥ 126 mg/dL
# Usamos distribuição normal com média 110 e desvio 35.
# clip() garante que os valores fiquem dentro do intervalo plausível.
glicose = np.random.normal(loc=110, scale=35, size=N)
glicose = np.clip(glicose, 60, 300).round(1)

# ─────────────────────────────────────────────────────────
# 6. GERAÇÃO DA VARIÁVEL: PRESSÃO ARTERIAL SISTÓLICA (mmHg)
# ─────────────────────────────────────────────────────────
# Referência clínica (sistólica):
#   Normal   : < 120 mmHg
#   Elevada  : 120 – 129 mmHg
#   HAS I    : 130 – 139 mmHg
#   HAS II   : ≥ 140 mmHg
# Média 125, desvio 20.
pressao_arterial = np.random.normal(loc=125, scale=20, size=N)
pressao_arterial = np.clip(pressao_arterial, 70, 220).round(1)

# ─────────────────────────────────────────────────────────
# 7. GERAÇÃO DA VARIÁVEL: IMC (kg/m²)
# ─────────────────────────────────────────────────────────
# Referência clínica:
#   Abaixo do peso : < 18.5
#   Normal         : 18.5 – 24.9
#   Sobrepeso      : 25.0 – 29.9
#   Obesidade I    : 30.0 – 34.9
#   Obesidade II   : 35.0 – 39.9
#   Obesidade III  : ≥ 40
# Média 27, desvio 6.
imc = np.random.normal(loc=27, scale=6, size=N)
imc = np.clip(imc, 14, 55).round(1)

# ─────────────────────────────────────────────────────────
# 8. GERAÇÃO DA VARIÁVEL: COLESTEROL TOTAL (mg/dL)
# ─────────────────────────────────────────────────────────
# Referência clínica:
#   Desejável  : < 200 mg/dL
#   Limítrofe  : 200 – 239 mg/dL
#   Alto       : ≥ 240 mg/dL
# Média 210, desvio 45.
colesterol = np.random.normal(loc=210, scale=45, size=N)
colesterol = np.clip(colesterol, 100, 400).round(1)

# ─────────────────────────────────────────────────────────
# 9. CÁLCULO DA VARIÁVEL-ALVO: RISCO (0, 1, 2)
# ─────────────────────────────────────────────────────────
# A classificação de risco é feita por pontuação acumulada.
# Cada fator de risco presente soma 1 ponto ao score do paciente.
# Ao final, o score total determina a classe de risco.
#
# Fatores de risco considerados:
#   1. Idade ≥ 50 anos
#   2. Glicose ≥ 126 mg/dL (diabetes)
#   3. Pressão arterial ≥ 140 mmHg (hipertensão estágio II)
#   4. IMC ≥ 30 (obesidade)
#   5. Colesterol ≥ 240 mg/dL (alto)

# Inicializa vetor de pontuação zerado
score = np.zeros(N, dtype=int)

# Incrementa 1 ponto para cada fator de risco presente
score += (idade >= 50).astype(int)           # Fator 1: idade
score += (glicose >= 126).astype(int)        # Fator 2: glicose
score += (pressao_arterial >= 140).astype(int)  # Fator 3: pressão
score += (imc >= 30).astype(int)             # Fator 4: IMC
score += (colesterol >= 240).astype(int)     # Fator 5: colesterol

# Converte o score em classe de risco:
#   0 pontos        → risco baixo  (0)
#   1 ou 2 pontos   → risco médio  (1)
#   3 ou mais       → risco alto   (2)
risco = np.where(score == 0, 0,
        np.where(score <= 2, 1, 2))

# ─────────────────────────────────────────────────────────
# 10. MONTAGEM DO DATAFRAME
# ─────────────────────────────────────────────────────────
# Organiza todas as colunas em um DataFrame do pandas.
df = pd.DataFrame({
    "nome":            nomes_pacientes,
    "idade":           idade,
    "glicose":         glicose,
    "pressao_arterial": pressao_arterial,
    "imc":             imc,
    "colesterol":      colesterol,
    "risco":           risco
})

# ─────────────────────────────────────────────────────────
# 11. EXPORTAÇÃO PARA CSV
# ─────────────────────────────────────────────────────────
# Salva o dataset em CSV sem incluir o índice do pandas.
caminho_csv = "pacientes.csv"
df.to_csv(caminho_csv, index=False, encoding="utf-8")
print(f"✅ Dataset gerado com sucesso! Arquivo salvo em: {caminho_csv}")

# ─────────────────────────────────────────────────────────
# 12. ANÁLISE EXPLORATÓRIA INICIAL (EDA BÁSICA)
# ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("       ANÁLISE EXPLORATÓRIA DO DATASET GERADO")
print("="*55)

print(f"\n📊 Dimensões do dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")

print("\n📋 Primeiros 5 registros:")
print(df.head().to_string(index=False))

print("\n📈 Estatísticas descritivas das variáveis numéricas:")
print(df.describe().round(2).to_string())

print("\n🎯 Distribuição da variável-alvo (risco):")
contagem = df["risco"].value_counts().sort_index()
rotulos  = {0: "Baixo Risco", 1: "Risco Médio", 2: "Risco Alto"}
for classe, count in contagem.items():
    pct = count / N * 100
    print(f"   Classe {classe} ({rotulos[classe]}): {count} pacientes ({pct:.1f}%)")

print("\n✅ Script finalizado com sucesso!\n")
