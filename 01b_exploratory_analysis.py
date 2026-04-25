# 01b_exploratory_analysis.py
# Analise Exploratoria das Series Temporais de SoC
# Esta etapa deve ser executada ANTES do ARIMA (Etapa 2)
# Fundamenta as escolhas metodologicas descritas na Secao 4 do artigo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ---------------------------------------------
# CONFIGURACOES
# ---------------------------------------------
DATA_PATH    = r"c:\Users\polia\Downloads\SoC_Experiment\data"
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)

DATASETS = {
    "CALCE" : "CALCE_synthetic_soc.csv",
    "OX"    : "OX_synthetic_soc.csv",
    "UL-PUR": "UL_PUR_synthetic_soc.csv",
    "SNL"   : "SNL_synthetic_soc.csv",
    "HNEI"  : "HNEI_synthetic_soc.csv",
}


# ---------------------------------------------
# FUNCAO: Carregamento simples (sem normalizacao)
# para visualizar os dados na escala original
# ---------------------------------------------
def load_raw(csv_filename, target_col="SoC"):
    filepath = os.path.join(DATA_PATH, csv_filename)
    df       = pd.read_csv(filepath)
    col_map  = {c.lower(): c for c in df.columns}
    real_col = col_map[target_col.lower()]
    return df[real_col].dropna().values.astype(float)


# ---------------------------------------------
# GRAFICO 1: Series brutas sobrepostas
# Mostra o comportamento geral de todas as series
# Para o relatorio: Figura de contexto dos dados
# ---------------------------------------------
def plot_all_series_overview(all_raw):
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=False)
    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]

    for i, (name, series) in enumerate(all_raw.items()):
        axes[i].plot(series, color=colors[i], linewidth=0.6)
        axes[i].set_title("[SINTETICO] {} - Serie SoC bruta ({} amostras)".format(
            name, len(series)))
        axes[i].set_ylabel("SoC")
        axes[i].set_xlabel("Passo de tempo")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Visao geral das series temporais de SoC por dataset",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "EDA_01_series_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafico salvo: {}".format(path))


# ---------------------------------------------
# GRAFICO 2: Estatisticas descritivas por dataset
# Para o relatorio: Tabela + boxplots
# ---------------------------------------------
def plot_boxplots(all_raw):
    fig, ax = plt.subplots(figsize=(12, 5))

    data_to_plot = [series for series in all_raw.values()]
    labels       = list(all_raw.keys())

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    notch=False, vert=True)

    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("[SINTETICO] Distribuicao dos valores de SoC por dataset")
    ax.set_ylabel("SoC")
    ax.set_xlabel("Dataset")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(RESULTS_PATH, "EDA_02_boxplots.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print("Grafico salvo: {}".format(path))


# ---------------------------------------------
# GRAFICO 3: Autocorrelacao (ACF) de cada serie
# Justifica o uso do ARIMA: se ha autocorrelacao,
# ha estrutura linear exploravel
# Para o relatorio: Evidencia de dependencia temporal
# ---------------------------------------------
def plot_autocorrelation(all_raw, max_lags=50):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    n_datasets = len(all_raw)
    fig, axes  = plt.subplots(n_datasets, 2, figsize=(14, 4 * n_datasets))

    for i, (name, series) in enumerate(all_raw.items()):
        plot_acf(series,  lags=max_lags, ax=axes[i, 0], title="ACF - {}".format(name))
        plot_pacf(series, lags=max_lags, ax=axes[i, 1], title="PACF - {}".format(name),
                  method='ywm')
        axes[i, 0].set_xlabel("Lag")
        axes[i, 1].set_xlabel("Lag")

    plt.suptitle("Funcoes de Autocorrelacao (ACF) e Autocorrelacao Parcial (PACF)",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "EDA_03_autocorrelation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafico salvo: {}".format(path))


# ---------------------------------------------
# GRAFICO 4: Primeiros N ciclos em detalhe
# Mostra claramente o padrao ciclico de carga/descarga
# Para o relatorio: Motivacao do problema
# ---------------------------------------------
def plot_detail_cycles(all_raw, n_samples=200):
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=False)
    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]

    for i, (name, series) in enumerate(all_raw.items()):
        segment = series[:n_samples]
        axes[i].plot(segment, color=colors[i], linewidth=1.2)
        axes[i].set_title("[SINTETICO] {} - Primeiros {} passos (detalhe dos ciclos)".format(
            name, n_samples))
        axes[i].set_ylabel("SoC")
        axes[i].set_xlabel("Passo de tempo")
        axes[i].grid(True, alpha=0.3)
        axes[i].fill_between(range(len(segment)), segment,
                             alpha=0.15, color=colors[i])

    plt.suptitle("Detalhe dos ciclos de carga/descarga - primeiros {} passos".format(
        n_samples), fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "EDA_04_cycle_detail.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("Grafico salvo: {}".format(path))


# ---------------------------------------------
# TABELA: Estatisticas descritivas
# Para o relatorio: Tabela de caracterizacao dos dados
# ---------------------------------------------
def print_descriptive_stats(all_raw):
    rows = []
    for name, series in all_raw.items():
        rows.append({
            "Dataset" : name,
            "N"       : len(series),
            "Min"     : round(series.min(), 4),
            "Max"     : round(series.max(), 4),
            "Media"   : round(series.mean(), 4),
            "Desvio"  : round(series.std(), 4),
            "Mediana" : round(np.median(series), 4),
        })

    df = pd.DataFrame(rows)
    print("\nEstatisticas descritivas das series de SoC (dados sinteticos):")
    print(df.to_string(index=False))
    print()

    path = os.path.join(RESULTS_PATH, "EDA_estatisticas_descritivas.csv")
    df.to_csv(path, index=False)
    print("Tabela salva em: {}".format(path))
    return df


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    print("=" * 55)
    print("  ETAPA 1b - ANALISE EXPLORATORIA (EDA)")
    print("  Fundamenta escolhas metodologicas do artigo")
    print("=" * 55)

    # Carrega todas as series brutas
    all_raw = {}
    for name, filename in DATASETS.items():
        try:
            all_raw[name] = load_raw(filename, target_col="SoC")
            print("Carregado: {}  ({} amostras)".format(name, len(all_raw[name])))
        except Exception as e:
            print("AVISO: Nao foi possivel carregar {}: {}".format(name, e))

    if not all_raw:
        print("Nenhum dataset carregado. Verifique o DATA_PATH.")
    else:
        # Estatisticas descritivas
        print_descriptive_stats(all_raw)

        # Graficos
        print("\nGerando graficos...")
        plot_all_series_overview(all_raw)
        plot_boxplots(all_raw)
        plot_autocorrelation(all_raw)
        plot_detail_cycles(all_raw, n_samples=200)

        print("\n" + "=" * 55)
        print("  EDA concluida!")
        print("  Graficos gerados em: results/EDA_*.png")
        print("  Proximo passo: 02_arima_stage.py")
        print("=" * 55)

        print("\nO QUE ANALISAR NO RELATORIO:")
        print("  1. ACF/PACF: barras significativas = dependencia temporal")
        print("     -> Justifica uso do ARIMA")
        print("  2. Detalhe dos ciclos: padrao nao-linear visivel")
        print("     -> Justifica estagio nao-linear (MLP/DTR/SVR)")
        print("  3. Boxplots: dispersao diferente por dataset")
        print("     -> Justifica selecao adaptativa de modelos")
        print("  4. Serie completa: tendencia de degradacao ao longo dos ciclos")
        print("     -> Justifica parametro d > 0 no ARIMA")