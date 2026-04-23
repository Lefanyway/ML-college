"""
========================================================
SISTEMA DE PREDIÇÃO DE RISCO CLÍNICO — INTERFACE CLI
Projeto: Machine Learning Aplicado à Saúde
Uso: python 03_preditor_cli.py
========================================================

Como funciona:
    1. O script treina automaticamente o modelo com o dataset
       'pacientes.csv' (gerado pelo Script 01).
    2. Entra em loop interativo: o usuário digita os dados
       de um paciente e recebe a análise de risco completa.
    3. É possível analisar vários pacientes em sequência.

Dependências:
    pandas, numpy, scikit-learn
    (as mesmas do projeto principal — sem nada extra)
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split

# ─────────────────────────────────────────────────────────
# UTILITÁRIOS DE TERMINAL
# ─────────────────────────────────────────────────────────

def limpar_tela():
    """Limpa o terminal (funciona em Windows e Linux/Mac)."""
    os.system("cls" if os.name == "nt" else "clear")


def linha(char="─", largura=58):
    print(char * largura)


def cabecalho():
    limpar_tela()
    linha("═")
    print("  🏥  SISTEMA DE PREDIÇÃO DE RISCO CLÍNICO  🏥")
    print("       Machine Learning Aplicado à Saúde")
    linha("═")
    print()


def colorir(texto, cor):
    """
    Aplica cor ANSI ao texto — funciona na maioria dos terminais.
    Windows 10+: funciona no Windows Terminal e PowerShell modernos.
    """
    cores = {
        "verde":    "\033[92m",
        "amarelo":  "\033[93m",
        "vermelho": "\033[91m",
        "ciano":    "\033[96m",
        "negrito":  "\033[1m",
        "reset":    "\033[0m",
    }
    return f"{cores.get(cor, '')}{texto}{cores['reset']}"


# ─────────────────────────────────────────────────────────
# BARRA DE PROGRESSO SIMPLES (para o treinamento)
# ─────────────────────────────────────────────────────────

def barra_progresso(mensagem, total=20):
    """Exibe uma barra de progresso animada enquanto treina."""
    print(f"\n  {mensagem}")
    print("  [", end="", flush=True)
    for i in range(total):
        print("█", end="", flush=True)
        # Pequena pausa simulada — o treino real roda em paralelo
        import time; time.sleep(0.04)
    print("] ✅\n")


# ─────────────────────────────────────────────────────────
# TREINAMENTO DO MODELO
# ─────────────────────────────────────────────────────────

def treinar_modelo(caminho_csv="pacientes.csv"):
    """
    Lê o dataset, treina o Random Forest e retorna
    o modelo treinado + o scaler ajustado.

    Retorna:
        modelo  : RandomForestClassifier treinado
        scaler  : StandardScaler ajustado no treino
        acuracia: float com a acurácia no conjunto de teste
    """
    # Verifica se o CSV existe
    if not os.path.exists(caminho_csv):
        print(colorir(f"\n  ❌ Arquivo '{caminho_csv}' não encontrado!", "vermelho"))
        print("  Execute primeiro o Script 01 (01_gerar_dataset.py)")
        print("  para gerar o dataset de treinamento.\n")
        sys.exit(1)

    df = pd.read_csv(caminho_csv)

    # Features e target
    X = df[["idade", "glicose", "pressao_arterial", "imc", "colesterol"]]
    y = df["risco"]

    # Divisão treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalização
    scaler = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)
    X_teste_norm  = scaler.transform(X_teste)

    # Treinamento
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino_norm, y_treino)

    acuracia = modelo.score(X_teste_norm, y_teste)

    return modelo, scaler, acuracia


# ─────────────────────────────────────────────────────────
# LEITURA DE INPUT COM VALIDAÇÃO
# ─────────────────────────────────────────────────────────

def ler_float(prompt, minimo, maximo, unidade=""):
    """
    Solicita um número float ao usuário com validação de range.
    Repete até receber um valor válido.

    Args:
        prompt  : texto exibido ao usuário
        minimo  : valor mínimo aceito
        maximo  : valor máximo aceito
        unidade : string da unidade (ex: "mg/dL") — só para mensagens

    Retorna:
        float válido dentro do intervalo [minimo, maximo]
    """
    while True:
        try:
            entrada = input(f"  {prompt}: ").strip().replace(",", ".")
            valor   = float(entrada)
            if minimo <= valor <= maximo:
                return valor
            else:
                print(colorir(
                    f"  ⚠  Valor fora do intervalo aceito: {minimo} a {maximo} {unidade}",
                    "amarelo"
                ))
        except ValueError:
            print(colorir("  ⚠  Digite apenas números (ex: 75 ou 23.5)", "amarelo"))


def ler_nome():
    """Lê o nome do paciente — permite qualquer string não vazia."""
    while True:
        nome = input("  Nome do paciente : ").strip()
        if nome:
            return nome
        print(colorir("  ⚠  Nome não pode ser vazio.", "amarelo"))


# ─────────────────────────────────────────────────────────
# EXIBIÇÃO DA ANÁLISE DE RISCO
# ─────────────────────────────────────────────────────────

def exibir_resultado(nome, dados, classe, probabilidades):
    """
    Formata e exibe o resultado completo da análise preditiva,
    incluindo:
      - Classificação de risco com emoji e cor
      - Probabilidades por classe (barra visual)
      - Análise individual de cada variável
      - Recomendação clínica
    """
    rotulos = {0: "BAIXO RISCO", 1: "RISCO MÉDIO", 2: "RISCO ALTO"}
    emojis  = {0: "🟢",          1: "🟡",          2: "🔴"}
    cores   = {0: "verde",       1: "amarelo",     2: "vermelho"}

    print()
    linha("═")
    print(f"  RESULTADO DA ANÁLISE — {nome.upper()}")
    linha("═")

    # ── Classificação principal ────────────────────────────
    resultado_texto = f"  {emojis[classe]}  {rotulos[classe]}"
    print(colorir(resultado_texto, cores[classe]))
    print(f"  Confiança do modelo: {probabilidades[classe]*100:.1f}%")
    linha()

    # ── Barra de probabilidades ───────────────────────────
    print("  Probabilidade por classe:\n")
    nomes_classe = ["Baixo Risco", "Risco Médio", "Risco Alto "]
    cores_classe = ["verde",       "amarelo",     "vermelho"  ]
    emojis_cls   = ["🟢",          "🟡",           "🔴"        ]

    for i, (n, c, e) in enumerate(zip(nomes_classe, cores_classe, emojis_cls)):
        pct   = probabilidades[i] * 100
        barras = int(pct / 3)          # escala: 100% = ~33 blocos
        barra  = "█" * barras + "░" * (33 - barras)
        linha_barra = f"  {e} {n} |{barra}| {pct:5.1f}%"
        print(colorir(linha_barra, c) if i == classe else linha_barra)

    linha()

    # ── Análise individual das variáveis ──────────────────
    print("  Análise dos fatores de risco:\n")

    def status_var(valor, limiar_medio, limiar_alto, unidade, nome_var, invertido=False):
        """
        Retorna emoji + texto de status para cada variável.
        invertido=True para variáveis onde valor baixo = risco (não usado aqui).
        """
        if valor >= limiar_alto:
            status = colorir(f"⚠  ALTO  ({valor} {unidade})", "vermelho")
        elif valor >= limiar_medio:
            status = colorir(f"~  ATENÇÃO ({valor} {unidade})", "amarelo")
        else:
            status = colorir(f"✓  NORMAL ({valor} {unidade})", "verde")
        print(f"    {nome_var:<22}: {status}")

    status_var(dados["idade"],            50,  65,  "anos",  "Idade          ")
    status_var(dados["glicose"],         100, 126,  "mg/dL", "Glicose        ")
    status_var(dados["pressao_arterial"],130, 140,  "mmHg",  "Pressão Arterial")
    status_var(dados["imc"],              25,  30,  "kg/m²", "IMC            ")
    status_var(dados["colesterol"],      200, 240,  "mg/dL", "Colesterol     ")

    linha()

    # ── Recomendação clínica ───────────────────────────────
    print("  Recomendação clínica:\n")

    if classe == 0:
        print(colorir("  ✅ Paciente com perfil de baixo risco.", "verde"))
        print("     → Manter estilo de vida saudável.")
        print("     → Consulta de rotina anual recomendada.")

    elif classe == 1:
        print(colorir("  ⚠️  Paciente com perfil de risco moderado.", "amarelo"))
        print("     → Agendar consulta médica nos próximos 30 dias.")
        print("     → Avaliar intervenção nos fatores alterados.")
        print("     → Monitoramento semestral recomendado.")

    else:
        print(colorir("  🚨 Paciente com perfil de ALTO RISCO!", "vermelho"))
        print("     → Encaminhamento médico urgente.")
        print("     → Avaliação especializada prioritária.")
        print("     → Não postergar intervenção clínica.")

    linha("═")

    # ── Aviso legal ───────────────────────────────────────
    aviso = "  ⚕  AVISO: Esta análise é gerada por IA e NÃO substitui"
    print(colorir(aviso, "ciano"))
    print(colorir("        diagnóstico médico profissional.", "ciano"))
    linha("═")
    print()


# ─────────────────────────────────────────────────────────
# COLETA DE DADOS DO PACIENTE
# ─────────────────────────────────────────────────────────

def coletar_dados_paciente():
    """
    Exibe os prompts de entrada e coleta as variáveis
    do paciente com validação de faixa para cada campo.

    Retorna:
        dict com nome e as 5 variáveis preditoras
    """
    print()
    linha()
    print("  📋  DADOS DO PACIENTE")
    linha()
    print("  Preencha os campos abaixo (use ponto ou vírgula")
    print("  para decimais). Pressione ENTER após cada valor.")
    print()

    nome             = ler_nome()
    idade            = ler_float("Idade (1–120 anos)",                     1,  120, "anos")
    glicose          = ler_float("Glicose em jejum (20–600 mg/dL)",       20,  600, "mg/dL")
    pressao_arterial = ler_float("Pressão arterial sistólica (40–300)",   40,  300, "mmHg")
    imc              = ler_float("IMC — Índice de Massa Corporal (10–80)", 10,   80, "kg/m²")
    colesterol       = ler_float("Colesterol total (50–600 mg/dL)",       50,  600, "mg/dL")

    return {
        "nome":             nome,
        "idade":            idade,
        "glicose":          glicose,
        "pressao_arterial": pressao_arterial,
        "imc":              imc,
        "colesterol":       colesterol,
    }


# ─────────────────────────────────────────────────────────
# REALIZAR A PREDIÇÃO
# ─────────────────────────────────────────────────────────

def prever(dados, modelo, scaler):
    """
    Aplica a normalização e faz a predição para um paciente.

    Args:
        dados  : dict com as variáveis do paciente
        modelo : RandomForestClassifier treinado
        scaler : StandardScaler ajustado

    Retorna:
        classe (int), probabilidades (array)
    """
    X = np.array([[
        dados["idade"],
        dados["glicose"],
        dados["pressao_arterial"],
        dados["imc"],
        dados["colesterol"],
    ]])

    X_norm        = scaler.transform(X)
    classe        = modelo.predict(X_norm)[0]
    probabilidades = modelo.predict_proba(X_norm)[0]

    return int(classe), probabilidades


# ─────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────

def main():
    cabecalho()

    # ── Carregamento e treinamento ─────────────────────────
    print("  Carregando dataset e treinando modelo...")
    barra_progresso("Treinando Random Forest com 100 árvores")

    modelo, scaler, acuracia = treinar_modelo("pacientes.csv")

    print(f"  Modelo treinado com acurácia de {colorir(f'{acuracia*100:.1f}%', 'verde')} no conjunto de teste.")
    print()

    # ── Loop de análise ───────────────────────────────────
    continuar = True
    while continuar:

        try:
            dados = coletar_dados_paciente()
        except KeyboardInterrupt:
            print(colorir("\n\n  Encerrando sistema. Até logo! 👋\n", "ciano"))
            sys.exit(0)

        print("\n  Processando análise preditiva...", end="", flush=True)
        import time; time.sleep(0.6)   # pausa dramática proposital 😄
        print(" ✅")

        classe, probabilidades = prever(dados, modelo, scaler)
        exibir_resultado(dados["nome"], dados, classe, probabilidades)

        # ── Perguntar se quer analisar outro paciente ─────
        while True:
            resp = input("  Analisar outro paciente? (s/n): ").strip().lower()
            if resp in ("s", "sim", "y", "yes"):
                cabecalho()
                print(f"  Modelo ativo | Acurácia: {colorir(f'{acuracia*100:.1f}%', 'verde')}")
                break
            elif resp in ("n", "nao", "não", "no"):
                continuar = False
                break
            else:
                print(colorir("  ⚠  Digite 's' para sim ou 'n' para não.", "amarelo"))

    print(colorir("\n  Sistema encerrado. Até logo! 👋\n", "ciano"))


# ─────────────────────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
