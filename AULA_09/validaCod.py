"""
Validação, Limpeza e Exploração do Dataset
Sistema de Predição de Risco Clínico - Sprint 3
Felipe Lemos de Moura & Guilherme da Silva Cruz
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CARREGAMENTO DO DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("pacientes.csv")

print("=" * 60)
print("  ETAPA 1 — VALIDAÇÃO E LIMPEZA DE DADOS")
print("=" * 60)

# ─────────────────────────────────────────────
# 1A. INJEÇÃO CONTROLADA DE PROBLEMAS (simulação realista)
# Injeta valores nulos e zeros em ~3% dos registros para
# demonstrar a rotina de limpeza em ambiente acadêmico.
# ─────────────────────────────────────────────
COLUNAS_CRITICAS = ["glicose", "pressao_arterial", "imc"]

np.random.seed(7)
for col in COLUNAS_CRITICAS:
    # ~2% de zeros inválidos
    idx_zeros = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[idx_zeros, col] = 0
    # ~1% de NaN
    idx_nulos = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[idx_nulos, col] = np.nan

# ─────────────────────────────────────────────
# 1B. DIAGNÓSTICO ANTES DA CORREÇÃO
# ─────────────────────────────────────────────
print("\n[1] DIAGNÓSTICO — Valores nulos e zeros inválidos detectados:\n")
print(f"  {'Coluna':<20} {'Nulos (NaN)':>12} {'Zeros (0)':>12} {'Total Inválidos':>16}")
print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*16}")

diagnostico = {}
for col in COLUNAS_CRITICAS:
    n_nulos  = df[col].isna().sum()
    n_zeros  = (df[col] == 0).sum()
    total    = n_nulos + n_zeros
    diagnostico[col] = {"nulos": n_nulos, "zeros": n_zeros, "total": total}
    print(f"  {col:<20} {n_nulos:>12} {n_zeros:>12} {total:>16}")

# ─────────────────────────────────────────────
# 1C. CORREÇÃO — Substituição pela MEDIANA
# Zeros inválidos são convertidos em NaN antes do fillna,
# garantindo que a mediana seja calculada sobre dados legítimos.
# ─────────────────────────────────────────────
print("\n[2] CORREÇÃO — Substituindo inválidos pela mediana da coluna:\n")
print(f"  {'Coluna':<20} {'Mediana Aplicada':>18} {'Registros Corrigidos':>22}")
print(f"  {'─'*20} {'─'*18} {'─'*22}")

for col in COLUNAS_CRITICAS:
    # Converte zeros para NaN para que não contaminem a mediana
    df[col] = df[col].replace(0, np.nan)

    # Calcula mediana somente com valores válidos
    mediana = df[col].median()

    # Conta quantos serão corrigidos
    n_corrigidos = df[col].isna().sum()

    # Aplica substituição
    df[col] = df[col].fillna(mediana)

    print(f"  {col:<20} {mediana:>18.2f} {n_corrigidos:>22}")

# ─────────────────────────────────────────────
# 1D. VERIFICAÇÃO PÓS-LIMPEZA
# ─────────────────────────────────────────────
print("\n[3] VERIFICAÇÃO PÓS-LIMPEZA — Nulos e zeros restantes:\n")
print(f"  {'Coluna':<20} {'Nulos (NaN)':>12} {'Zeros (0)':>12}  {'Status':>10}")
print(f"  {'─'*20} {'─'*12} {'─'*12}  {'─'*10}")

tudo_ok = True
for col in COLUNAS_CRITICAS:
    n_nulos = df[col].isna().sum()
    n_zeros = (df[col] == 0).sum()
    status  = "✅ OK" if (n_nulos == 0 and n_zeros == 0) else "⚠️  REVISAR"
    if status != "✅ OK":
        tudo_ok = False
    print(f"  {col:<20} {n_nulos:>12} {n_zeros:>12}  {status:>10}")

print(f"\n  Resultado geral: {'✅ Dataset limpo e pronto para uso.' if tudo_ok else '⚠️  Ainda há inconsistências.'}")

# Salva o dataset limpo
df.to_csv("pacientes_limpo.csv", index=False, encoding="utf-8")
print("\n  💾 Dataset corrigido salvo em: pacientes_limpo.csv")

# ─────────────────────────────────────────────
# ETAPA 2 — EXPLORAÇÃO: TOP 10 RANKINGS
# ─────────────────────────────────────────────
print("\n")
print("=" * 60)
print("  ETAPA 2 — EXPLORAÇÃO: RANKINGS CLÍNICOS")
print("=" * 60)

RISCO_LABEL = {0: "Baixo", 1: "Médio", 2: "Alto"}

def exibir_top10(df, coluna, titulo, unidade):
    top = (
        df.nlargest(10, coluna)
          .reset_index(drop=True)
    )
    top.index += 1  # ranking começa em 1

    print(f"\n{'─'*60}")
    print(f"  🏆 TOP 10 — {titulo} ({unidade})")
    print(f"{'─'*60}")
    print(f"  {'#':>2}  {'Nome':<14} {'Idade':>5}  {coluna.capitalize():>10}  {'IMC':>6}  {'Risco':>6}")
    print(f"  {'─'*2}  {'─'*14} {'─'*5}  {'─'*10}  {'─'*6}  {'─'*6}")

    for rank, row in top.iterrows():
        risco_txt = RISCO_LABEL[int(row["risco"])]
        print(
            f"  {rank:>2}  {row['nome']:<14} {int(row['idade']):>5}  "
            f"{row[coluna]:>10.2f}  {row['imc']:>6.2f}  {risco_txt:>6}"
        )

    # Estatísticas rápidas do top 10
    media_col   = top[coluna].mean()
    media_idade = top["idade"].mean()
    pct_alto    = (top["risco"] == 2).sum() / 10 * 100

    print(f"\n  📊 Resumo do grupo:")
    print(f"     Média de {coluna:<18}: {media_col:.2f} {unidade}")
    print(f"     Média de idade          : {media_idade:.1f} anos")
    print(f"     Pacientes em Alto Risco : {pct_alto:.0f}%")

exibir_top10(df, "glicose",    "MAIORES NÍVEIS DE GLICOSE",    "mg/dL")
exibir_top10(df, "colesterol", "MAIORES NÍVEIS DE COLESTEROL", "mg/dL")

print(f"\n{'='*60}\n")