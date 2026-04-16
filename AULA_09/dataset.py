"""
Gerador de Dataset Sintético - Sistema de Predição de Risco Clínico
Sprint 3 - Felipe Lemos de Moura & Guilherme da Silva Cruz
"""

import numpy as np
import pandas as pd

# Semente para reprodutibilidade
np.random.seed(42)

N = 2000

# ─────────────────────────────────────────────
# 1. NOMES FICTÍCIOS
# ─────────────────────────────────────────────
nomes_masculinos = [
    "Carlos", "João", "Pedro", "Lucas", "Marcos", "Rafael", "Bruno",
    "Diego", "André", "Felipe", "Rodrigo", "Eduardo", "Gustavo", "Thiago",
    "Leonardo", "Mateus", "Gabriel", "Henrique", "Fernando", "Paulo",
    "Renato", "Vitor", "Leandro", "Caio", "Murilo", "Danilo", "Fábio",
    "Sérgio", "Márcio", "Roberto", "Luiz", "Julio", "Claudio", "Alexandro"
]

nomes_femininos = [
    "Ana", "Maria", "Julia", "Fernanda", "Camila", "Beatriz", "Amanda",
    "Larissa", "Leticia", "Patricia", "Vanessa", "Aline", "Bruna", "Carla",
    "Daniela", "Elisa", "Gabriela", "Helena", "Isabela", "Juliana",
    "Karen", "Lucia", "Mariana", "Natalia", "Olivia", "Priscila",
    "Renata", "Sandra", "Tatiana", "Ursula", "Vera", "Yasmin", "Zelia"
]

todos_nomes = nomes_masculinos + nomes_femininos
nomes = np.random.choice(todos_nomes, size=N)

# ─────────────────────────────────────────────
# 2. VARIÁVEIS CLÍNICAS COM FAIXAS REALISTAS
# ─────────────────────────────────────────────

# Idade: 18–99, distribuição levemente enviesada para adultos maduros
idade = np.random.beta(a=2.5, b=2.0, size=N) * (99 - 18) + 18
idade = np.round(idade).astype(int)

# Glicose (mg/dL): normal 70–99 | pré-diabético 100–125 | diabético ≥126
glicose = np.random.normal(loc=105, scale=30, size=N)
glicose = np.clip(glicose, 60, 350).round(2)

# Pressão Arterial Sistólica (mmHg): normal <120 | elevada 120-129 | hipertensão ≥130
pressao_arterial = np.random.normal(loc=125, scale=20, size=N)
pressao_arterial = np.clip(pressao_arterial, 70, 220).round(2)

# IMC (kg/m²): baixo peso <18.5 | normal 18.5-24.9 | sobrepeso 25-29.9 | obeso ≥30
imc = np.random.normal(loc=27, scale=5.5, size=N)
imc = np.clip(imc, 14.0, 55.0).round(2)

# Colesterol Total (mg/dL): desejável <200 | limítrofe 200-239 | alto ≥240
colesterol = np.random.normal(loc=210, scale=45, size=N)
colesterol = np.clip(colesterol, 100, 400).round(2)

# ─────────────────────────────────────────────
# 3. REGRA DE RISCO CLÍNICO (score baseado em faixas)
# ─────────────────────────────────────────────
# Cada variável contribui com pontos (0, 1 ou 2) conforme severidade clínica.
# Score total 0–10 → classifica em Baixo (0), Médio (1) ou Alto (2)

def pontuar_glicose(g):
    if g < 100:
        return 0
    elif g < 126:
        return 1
    else:
        return 2

def pontuar_pressao(p):
    if p < 120:
        return 0
    elif p < 140:
        return 1
    else:
        return 2

def pontuar_imc(i):
    if i < 25:
        return 0
    elif i < 30:
        return 1
    else:
        return 2

def pontuar_colesterol(c):
    if c < 200:
        return 0
    elif c < 240:
        return 1
    else:
        return 2

def pontuar_idade(a):
    if a < 35:
        return 0
    elif a < 55:
        return 1
    else:
        return 2

# Vetorização
vp_glicose      = np.vectorize(pontuar_glicose)
vp_pressao      = np.vectorize(pontuar_pressao)
vp_imc          = np.vectorize(pontuar_imc)
vp_colesterol   = np.vectorize(pontuar_colesterol)
vp_idade        = np.vectorize(pontuar_idade)

score = (
    vp_glicose(glicose) * 2.5 +       # glicose: peso maior (diagnóstico direto)
    vp_pressao(pressao_arterial) * 2.0 +
    vp_imc(imc) * 1.5 +
    vp_colesterol(colesterol) * 2.0 +
    vp_idade(idade) * 2.0
)
# Score máximo possível: 2*2.5 + 2*2.0 + 2*1.5 + 2*2.0 + 2*2.0 = 20

# Ruído leve para evitar fronteiras artificiais perfeitas
score += np.random.normal(0, 0.5, size=N)
score = np.clip(score, 0, 20)

# Classificação: Baixo (0-7), Médio (7-13), Alto (>13)
risco = np.where(score < 7, 0, np.where(score < 13, 1, 2))

# ─────────────────────────────────────────────
# 4. MONTAGEM E SALVAMENTO DO DATAFRAME
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "nome":             nomes,
    "idade":            idade,
    "glicose":          glicose,
    "pressao_arterial": pressao_arterial,
    "imc":              imc,
    "colesterol":       colesterol,
    "risco":            risco
})

df.to_csv("pacientes.csv", index=False, encoding="utf-8")

# ─────────────────────────────────────────────
# 5. RELATÓRIO DE VALIDAÇÃO
# ─────────────────────────────────────────────
print("=" * 55)
print("  DATASET GERADO COM SUCESSO: pacientes.csv")
print("=" * 55)
print(f"\nTotal de registros : {len(df)}")
print(f"\n{'─'*40}")
print("DISTRIBUIÇÃO DA VARIÁVEL ALVO (risco):")
contagem = df["risco"].value_counts().sort_index()
labels   = {0: "Baixo Risco (0)", 1: "Médio Risco (1)", 2: "Alto Risco  (2)"}
for k, v in contagem.items():
    pct = v / N * 100
    print(f"  {labels[k]}: {v:5d} registros  ({pct:.1f}%)")

print(f"\n{'─'*40}")
print("ESTATÍSTICAS DESCRITIVAS DAS FEATURES:\n")
print(df[["idade","glicose","pressao_arterial","imc","colesterol"]].describe().round(2).to_string())
print("\n" + "=" * 55)