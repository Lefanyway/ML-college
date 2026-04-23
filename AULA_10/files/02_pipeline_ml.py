"""
========================================================
SCRIPT 02 — PIPELINE COMPLETO DE MACHINE LEARNING
Projeto: Sistema de Predição de Risco Clínico
Disciplina: Machine Learning Aplicado à Saúde
========================================================

Objetivo:
    Treinar, avaliar e comparar múltiplos modelos de ML para
    prever o nível de risco clínico de um paciente com base
    em variáveis biomédicas.

Pipeline executado:
    1. Leitura do dataset
    2. Análise exploratória
    3. Pré-processamento (separação X/y, treino/teste, normalização)
    4. Treinamento de múltiplos modelos
    5. Avaliação com métricas (acurácia, precision, recall, F1)
    6. Validação cruzada (K-Fold, k=5)
    7. Visualizações (comparação de modelos, matriz de confusão, ROC)
    8. Predição de novo paciente

Bibliotecas utilizadas:
    pandas, numpy, scikit-learn, matplotlib
"""

# ─────────────────────────────────────────────────────────
# IMPORTAÇÕES — todas as bibliotecas necessárias
# ─────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")  # Suprime avisos de convergência

# Pré-processamento
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ═══════════════════════════════════════════════════════════
# BLOCO 1 — LEITURA E INSPEÇÃO DO DATASET
# ═══════════════════════════════════════════════════════════

print("="*60)
print("   BLOCO 1 — LEITURA DO DATASET")
print("="*60)

# Lê o CSV gerado pelo Script 01
df = pd.read_csv("pacientes.csv")

print(f"\n✅ Dataset carregado com {df.shape[0]} registros e {df.shape[1]} colunas.")
print("\n📋 Primeiros registros:")
print(df.head().to_string(index=False))

print("\n📊 Distribuição da variável-alvo:")
print(df["risco"].value_counts().sort_index().rename({0: "Baixo", 1: "Médio", 2: "Alto"}))

print("\n🔍 Verificação de valores ausentes:")
print(df.isnull().sum())

# ═══════════════════════════════════════════════════════════
# BLOCO 2 — PRÉ-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 2 — PRÉ-PROCESSAMENTO")
print("="*60)

# ── 2.1 Separação de features (X) e variável-alvo (y) ──────
# A coluna 'nome' é textual e não tem valor preditivo → removida.
# 'risco' é o que queremos prever → separada como y.
X = df.drop(columns=["nome", "risco"])   # features preditoras
y = df["risco"]                          # variável-alvo

print(f"\n✅ Features (X): {list(X.columns)}")
print(f"✅ Variável-alvo (y): 'risco' | Classes: {sorted(y.unique())}")

# ── 2.2 Divisão em treino e teste ─────────────────────────
# 80% para treino, 20% para teste.
# stratify=y garante proporção igual das classes nos dois subconjuntos.
# random_state assegura reprodutibilidade.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📂 Tamanho do conjunto de treino : {X_treino.shape[0]} amostras")
print(f"📂 Tamanho do conjunto de teste  : {X_teste.shape[0]} amostras")

# ── 2.3 Normalização com StandardScaler ───────────────────
# Converte cada feature para média 0 e desvio padrão 1.
# IMPORTANTE: o scaler é ajustado APENAS no treino (fit),
# e aplicado (transform) tanto no treino quanto no teste.
# Isso evita data leakage (vazamento de informação do teste).
scaler = StandardScaler()

X_treino_norm = scaler.fit_transform(X_treino)   # aprende + transforma
X_teste_norm  = scaler.transform(X_teste)         # só transforma

print("\n✅ Normalização aplicada com StandardScaler.")
print("   Médias antes da normalização (treino):", X_treino.mean().round(2).values)
print("   Médias após  a normalização (treino):", X_treino_norm.mean(axis=0).round(4))

# ═══════════════════════════════════════════════════════════
# BLOCO 3 — DEFINIÇÃO E TREINAMENTO DOS MODELOS
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 3 — TREINAMENTO DOS MODELOS")
print("="*60)

# Dicionário com os três modelos — facilita iterar e comparar
modelos = {
    "Regressão Logística": LogisticRegression(
        max_iter=1000,         # número máximo de iterações para convergir
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,      # 100 árvores de decisão no ensemble
        max_depth=None,        # árvores crescem até separação perfeita
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=7,         # considera os 7 vizinhos mais próximos
        weights="distance"     # vizinhos mais próximos têm mais peso
    )
}

# Dicionário para armazenar os resultados de cada modelo
resultados = {}

# ── Loop de treinamento e avaliação básica ─────────────────
for nome_modelo, modelo in modelos.items():
    print(f"\n🤖 Treinando: {nome_modelo}...")

    # Treina o modelo com os dados normalizados de treino
    modelo.fit(X_treino_norm, y_treino)

    # Faz predições no conjunto de teste
    y_pred = modelo.predict(X_teste_norm)

    # ── Métricas de avaliação ──────────────────────────────
    # accuracy: fração de predições corretas
    acc = accuracy_score(y_teste, y_pred)

    # precision: das predições positivas, quantas são corretas
    # average='weighted': pondera pela frequência de cada classe
    prec = precision_score(y_teste, y_pred, average="weighted", zero_division=0)

    # recall: das amostras positivas reais, quantas foram capturadas
    rec  = recall_score(y_teste, y_pred, average="weighted", zero_division=0)

    # f1: média harmônica entre precision e recall
    f1   = f1_score(y_teste, y_pred, average="weighted", zero_division=0)

    # Armazena tudo no dicionário de resultados
    resultados[nome_modelo] = {
        "modelo":   modelo,
        "y_pred":   y_pred,
        "acuracia": acc,
        "precision": prec,
        "recall":   rec,
        "f1":       f1
    }

    print(f"   Acurácia : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")

# ═══════════════════════════════════════════════════════════
# BLOCO 4 — VALIDAÇÃO CRUZADA (K-FOLD, k=5)
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 4 — VALIDAÇÃO CRUZADA (K-Fold, k=5)")
print("="*60)

print("""
  O que é K-Fold Cross Validation?
  ─────────────────────────────────
  O dataset de treino é dividido em k=5 partes iguais (folds).
  O modelo é treinado k vezes: em cada rodada, 4 folds são usados
  para treino e 1 fold para validação. Isso garante avaliação mais
  robusta e menos dependente de uma única divisão aleatória.
""")

# StratifiedKFold garante que cada fold mantenha a proporção das classes
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Juntar treino e teste normalizados para a validação cruzada completa
X_completo_norm = np.vstack([X_treino_norm, X_teste_norm])
y_completo      = pd.concat([y_treino, y_teste]).reset_index(drop=True)

for nome_modelo, info in resultados.items():
    modelo = info["modelo"]

    # cross_val_score retorna um array com a acurácia de cada fold
    scores = cross_val_score(
        modelo,
        X_completo_norm,
        y_completo,
        cv=kfold,
        scoring="accuracy"
    )

    resultados[nome_modelo]["cv_scores"] = scores
    resultados[nome_modelo]["cv_media"]  = scores.mean()
    resultados[nome_modelo]["cv_std"]    = scores.std()

    print(f"\n  {nome_modelo}:")
    print(f"    Scores por fold: {[f'{s:.4f}' for s in scores]}")
    print(f"    Média  : {scores.mean():.4f}")
    print(f"    Desvio : {scores.std():.4f}")

# ═══════════════════════════════════════════════════════════
# BLOCO 5 — COMPARAÇÃO E SELEÇÃO DO MELHOR MODELO
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 5 — COMPARAÇÃO DOS MODELOS")
print("="*60)

# Monta uma tabela comparativa com todas as métricas
linhas = []
for nome, info in resultados.items():
    linhas.append({
        "Modelo":           nome,
        "Acurácia":         round(info["acuracia"],  4),
        "Precision":        round(info["precision"], 4),
        "Recall":           round(info["recall"],    4),
        "F1-Score":         round(info["f1"],        4),
        "CV Média":         round(info["cv_media"],  4),
        "CV Desvio":        round(info["cv_std"],    4)
    })

df_resultados = pd.DataFrame(linhas).set_index("Modelo")
print("\n📊 Tabela comparativa completa:")
print(df_resultados.to_string())

# Identifica o melhor modelo pela média da validação cruzada
melhor_nome = max(resultados, key=lambda k: resultados[k]["cv_media"])
melhor_info = resultados[melhor_nome]

print(f"\n🏆 Melhor modelo: {melhor_nome}")
print(f"   CV Média: {melhor_info['cv_media']:.4f} ± {melhor_info['cv_std']:.4f}")

# Relatório detalhado do melhor modelo
print(f"\n📋 Relatório completo do melhor modelo ({melhor_nome}):")
print(classification_report(
    y_teste,
    melhor_info["y_pred"],
    target_names=["Baixo Risco", "Risco Médio", "Risco Alto"]
))

# ═══════════════════════════════════════════════════════════
# BLOCO 6 — VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 6 — GERANDO VISUALIZAÇÕES")
print("="*60)

# Paleta de cores para os modelos
cores_modelos = {
    "Regressão Logística": "#4C72B0",
    "Random Forest":       "#55A868",
    "KNN":                 "#C44E52"
}

# ─────────────────────────────────────────────────────────
# FIGURA 1 — Comparação de Métricas entre Modelos
# ─────────────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Comparação de Desempenho dos Modelos", fontsize=14, fontweight="bold", y=1.02)

nomes_m  = list(resultados.keys())
acuracias = [resultados[n]["acuracia"] for n in nomes_m]
f1s       = [resultados[n]["f1"]       for n in nomes_m]
cv_medias = [resultados[n]["cv_media"] for n in nomes_m]
cv_stds   = [resultados[n]["cv_std"]   for n in nomes_m]
cores_lista = [cores_modelos[n] for n in nomes_m]

# Gráfico 1a — Comparação de Acurácia, F1 e CV
ax1 = axes[0]
x    = np.arange(len(nomes_m))
larg = 0.25  # largura de cada barra

ax1.bar(x - larg, acuracias, larg, label="Acurácia (teste)",  color="#4C72B0", alpha=0.85)
ax1.bar(x,         f1s,       larg, label="F1-Score (teste)", color="#55A868", alpha=0.85)
ax1.bar(x + larg,  cv_medias, larg, label="CV Média (k=5)",   color="#C44E52", alpha=0.85,
        yerr=cv_stds, capsize=5)

ax1.set_title("Métricas por Modelo", fontsize=12)
ax1.set_ylabel("Score")
ax1.set_xticks(x)
ax1.set_xticklabels(nomes_m, rotation=10, fontsize=9)
ax1.set_ylim(0, 1.1)
ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.4)

# Gráfico 1b — Tabela visual de métricas
ax2 = axes[1]
ax2.axis("off")

# Cria os dados da tabela
colunas_tab = ["Acurácia", "Precision", "Recall", "F1", "CV Média", "CV Std"]
dados_tab   = [
    [
        f"{resultados[n]['acuracia']:.3f}",
        f"{resultados[n]['precision']:.3f}",
        f"{resultados[n]['recall']:.3f}",
        f"{resultados[n]['f1']:.3f}",
        f"{resultados[n]['cv_media']:.3f}",
        f"{resultados[n]['cv_std']:.4f}"
    ]
    for n in nomes_m
]

tabela = ax2.table(
    cellText=dados_tab,
    rowLabels=nomes_m,
    colLabels=colunas_tab,
    cellLoc="center",
    loc="center"
)
tabela.auto_set_font_size(False)
tabela.set_fontsize(9)
tabela.scale(1.1, 2.0)

# Destaca o melhor modelo na tabela (linha verde)
idx_melhor = nomes_m.index(melhor_nome)
for col in range(len(colunas_tab)):
    tabela[idx_melhor + 1, col].set_facecolor("#d4edda")  # verde claro

ax2.set_title("Tabela de Resultados\n(destaque = melhor modelo)", fontsize=11)

plt.tight_layout()
plt.savefig("fig1_comparacao_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figura 1 salva: fig1_comparacao_modelos.png")

# ─────────────────────────────────────────────────────────
# FIGURA 2 — Matriz de Confusão do Melhor Modelo
# ─────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(7, 5))

cm      = confusion_matrix(y_teste, melhor_info["y_pred"])
classes = ["Baixo Risco", "Risco Médio", "Risco Alto"]

# Exibe a matriz com escala de cor
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

# Adiciona os rótulos dos eixos
ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    title=f"Matriz de Confusão — {melhor_nome}",
    ylabel="Classe Real",
    xlabel="Classe Predita"
)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Escreve os valores dentro de cada célula da matriz
limiar = cm.max() / 2.0
for i in range(len(classes)):
    for j in range(len(classes)):
        cor_texto = "white" if cm[i, j] > limiar else "black"
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=cor_texto)

plt.tight_layout()
plt.savefig("fig2_matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figura 2 salva: fig2_matriz_confusao.png")

# ─────────────────────────────────────────────────────────
# FIGURA 3 — Curvas ROC (One-vs-Rest) do Melhor Modelo
# ─────────────────────────────────────────────────────────
# ROC avalia a capacidade de separação para cada classe.
# One-vs-Rest: cada classe é tratada como "positivo" vs "todo o resto".

fig3, ax = plt.subplots(figsize=(8, 6))

classes_bin = [0, 1, 2]
y_teste_bin = label_binarize(y_teste, classes=classes_bin)  # converte para formato binário

# Verifica se o modelo suporta predict_proba
if hasattr(melhor_info["modelo"], "predict_proba"):
    y_prob = melhor_info["modelo"].predict_proba(X_teste_norm)

    cores_roc    = ["#4C72B0", "#55A868", "#C44E52"]
    nomes_classe = ["Baixo Risco (0)", "Risco Médio (1)", "Risco Alto (2)"]

    for i, (cor, nome_c) in enumerate(zip(cores_roc, nomes_classe)):
        fpr, tpr, _ = roc_curve(y_teste_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cor, lw=2,
                label=f"{nome_c} (AUC = {roc_auc:.3f})")

    # Linha diagonal = classificador aleatório (baseline)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC = 0.5)")

    ax.set(
        title=f"Curvas ROC (One-vs-Rest) — {melhor_nome}",
        xlabel="Taxa de Falsos Positivos (FPR)",
        ylabel="Taxa de Verdadeiros Positivos (TPR)",
        xlim=[0, 1],
        ylim=[0, 1.02]
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig3_curvas_roc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura 3 salva: fig3_curvas_roc.png")
else:
    print(f"⚠️  O modelo {melhor_nome} não suporta predict_proba. Curva ROC pulada.")

# ─────────────────────────────────────────────────────────
# FIGURA 4 — Importância das Features (Random Forest)
# ─────────────────────────────────────────────────────────
# O Random Forest calcula a importância de cada feature
# com base na redução de impureza (Gini) que ela proporciona.

if "Random Forest" in resultados:
    fig4, ax = plt.subplots(figsize=(8, 4))
    rf_model     = resultados["Random Forest"]["modelo"]
    importancias = rf_model.feature_importances_
    features     = list(X.columns)

    # Ordena da maior para a menor importância
    idx_ord = np.argsort(importancias)[::-1]

    ax.bar(
        [features[i] for i in idx_ord],
        importancias[idx_ord],
        color="#55A868",
        edgecolor="black",
        alpha=0.85
    )
    ax.set_title("Importância das Features — Random Forest", fontsize=12)
    ax.set_ylabel("Importância (Gini)")
    ax.set_xlabel("Feature")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig4_importancia_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura 4 salva: fig4_importancia_features.png")

# ═══════════════════════════════════════════════════════════
# BLOCO 7 — PREDIÇÃO DE NOVO PACIENTE
# ═══════════════════════════════════════════════════════════

print("\n" + "="*60)
print("   BLOCO 7 — PREDIÇÃO DE NOVO PACIENTE")
print("="*60)

# ── Simulação de um novo paciente ─────────────────────────
# Estes valores representam um paciente hipotético com
# múltiplos fatores de risco (alta glicose, IMC alto, etc.)
novo_paciente = {
    "nome":             "Carlos",
    "idade":            62,
    "glicose":          148.0,    # acima de 126 → diabetes
    "pressao_arterial": 155.0,    # acima de 140 → HAS II
    "imc":              34.2,     # acima de 30  → obesidade
    "colesterol":       258.0     # acima de 240 → alto
}

print(f"\n👤 Novo paciente: {novo_paciente['nome']}")
print(f"   Idade            : {novo_paciente['idade']} anos")
print(f"   Glicose          : {novo_paciente['glicose']} mg/dL")
print(f"   Pressão Arterial : {novo_paciente['pressao_arterial']} mmHg")
print(f"   IMC              : {novo_paciente['imc']} kg/m²")
print(f"   Colesterol       : {novo_paciente['colesterol']} mg/dL")

# Monta array com as features do novo paciente (na mesma ordem do treino)
X_novo = np.array([[
    novo_paciente["idade"],
    novo_paciente["glicose"],
    novo_paciente["pressao_arterial"],
    novo_paciente["imc"],
    novo_paciente["colesterol"]
]])

# Aplica a mesma normalização usada no treino
X_novo_norm = scaler.transform(X_novo)

# ── Predição com o melhor modelo ───────────────────────────
melhor_modelo = melhor_info["modelo"]

# Classe predita (0, 1 ou 2)
classe_pred = melhor_modelo.predict(X_novo_norm)[0]

# Probabilidades para cada classe
if hasattr(melhor_modelo, "predict_proba"):
    probabilidades = melhor_modelo.predict_proba(X_novo_norm)[0]
else:
    probabilidades = None

# Mapeamento de classe para rótulo legível
rotulo_risco = {0: "BAIXO RISCO", 1: "RISCO MÉDIO", 2: "RISCO ALTO"}
emoji_risco  = {0: "🟢",          1: "🟡",           2: "🔴"}

print("\n" + "─"*50)
print(f"  🏆 Modelo utilizado: {melhor_nome}")
print("─"*50)

print(f"\n  {emoji_risco[classe_pred]}  Classificação prevista: {rotulo_risco[classe_pred]}")

if probabilidades is not None:
    print("\n  📊 Probabilidades por classe:")
    print(f"     🟢 Baixo Risco : {probabilidades[0]*100:.1f}%")
    print(f"     🟡 Risco Médio : {probabilidades[1]*100:.1f}%")
    print(f"     🔴 Risco Alto  : {probabilidades[2]*100:.1f}%")
    confianca = probabilidades[classe_pred] * 100
    print(f"\n  🎯 Confiança na predição: {confianca:.1f}%")

print("\n" + "─"*50)

# Aviso clínico baseado na classificação
if classe_pred == 0:
    print("  ✅ Paciente com perfil de baixo risco clínico.")
    print("     Recomendação: manter acompanhamento preventivo regular.")
elif classe_pred == 1:
    print("  ⚠️  Paciente com perfil de risco moderado.")
    print("     Recomendação: acompanhamento médico periódico e")
    print("     avaliação dos fatores de risco identificados.")
else:
    print("  🚨 Paciente com perfil de alto risco clínico!")
    print("     Recomendação: encaminhamento urgente para avaliação")
    print("     médica especializada e plano de intervenção imediato.")

print("\n" + "═"*60)
print("   PIPELINE CONCLUÍDO COM SUCESSO!")
print("═"*60)
print("\n📁 Arquivos gerados:")
print("   pacientes.csv                   → dataset completo")
print("   fig1_comparacao_modelos.png     → métricas comparativas")
print("   fig2_matriz_confusao.png        → matriz de confusão")
print("   fig3_curvas_roc.png             → curvas ROC")
print("   fig4_importancia_features.png   → importância das features")
print()
