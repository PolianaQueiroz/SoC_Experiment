# 07_nemenyi_test.py
# Teste estatistico de Nemenyi — equivalente a Figura 4 do artigo
# Referencia: Demsar [35], significancia alpha = 0.05
# Determina se as diferencas de desempenho entre os modelos
# sao estatisticamente significativas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
import os

RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# ---------------------------------------------
# DADOS: MSE de todos os modelos em todos os
# datasets — coletados das etapas anteriores
# ---------------------------------------------
# Linhas = modelos | Colunas = datasets
MSE_DATA = {
    "CALCE" : {
        "ARIMA"         : 0.007146,
        "MLP"           : 0.008768,
        "SVR"           : 0.008115,
        "DTR"           : 0.011801,
        "ARIMA+MLP"     : 0.007198,
        "ARIMA+SVR"     : 0.007146,
        "SVR(ARIMA,SVR)": 0.007530,
        "Proposto"      : 0.007573,
    },
    "OX" : {
        "ARIMA"         : 0.006982,
        "MLP"           : 0.011947,
        "SVR"           : 0.009028,
        "DTR"           : 0.027617,
        "ARIMA+MLP"     : 0.006996,
        "ARIMA+SVR"     : 0.006976,
        "SVR(ARIMA,SVR)": 0.007032,
        "Proposto"      : 0.017368,
    },
    "UL-PUR" : {
        "ARIMA"         : 0.008217,
        "MLP"           : 0.015738,
        "SVR"           : 0.012829,
        "DTR"           : 0.028293,
        "ARIMA+MLP"     : 0.008236,
        "ARIMA+SVR"     : 0.008221,
        "SVR(ARIMA,SVR)": 0.008304,
        "Proposto"      : 0.008249,
    },
    "SNL" : {
        "ARIMA"         : 0.005199,
        "MLP"           : 0.011306,
        "SVR"           : 0.005968,
        "DTR"           : 0.024403,
        "ARIMA+MLP"     : 0.005169,
        "ARIMA+SVR"     : 0.005183,
        "SVR(ARIMA,SVR)": 0.005201,
        "Proposto"      : 0.005183,
    },
    "HNEI" : {
        "ARIMA"         : 0.009545,
        "MLP"           : 0.014115,
        "SVR"           : 0.014830,
        "DTR"           : 0.033797,
        "ARIMA+MLP"     : 0.009541,
        "ARIMA+SVR"     : 0.009528,
        "SVR(ARIMA,SVR)": 0.009633,
        "Proposto"      : 0.009541,
    },
}

MAE_DATA = {
    "CALCE" : {
        "ARIMA"         : 0.066373,
        "MLP"           : 0.073515,
        "SVR"           : 0.070632,
        "DTR"           : 0.089400,
        "ARIMA+MLP"     : 0.066540,
        "ARIMA+SVR"     : 0.066376,
        "SVR(ARIMA,SVR)": 0.067955,
        "Proposto"      : 0.068155,
    },
    "OX" : {
        "ARIMA"         : 0.066689,
        "MLP"           : 0.087882,
        "SVR"           : 0.075728,
        "DTR"           : 0.141142,
        "ARIMA+MLP"     : 0.066960,
        "ARIMA+SVR"     : 0.066798,
        "SVR(ARIMA,SVR)": 0.066785,
        "Proposto"      : 0.102772,
    },
    "UL-PUR" : {
        "ARIMA"         : 0.072233,
        "MLP"           : 0.103602,
        "SVR"           : 0.092021,
        "DTR"           : 0.143308,
        "ARIMA+MLP"     : 0.072342,
        "ARIMA+SVR"     : 0.072221,
        "SVR(ARIMA,SVR)": 0.073029,
        "Proposto"      : 0.072689,
    },
    "SNL" : {
        "ARIMA"         : 0.057654,
        "MLP"           : 0.088078,
        "SVR"           : 0.062780,
        "DTR"           : 0.134502,
        "ARIMA+MLP"     : 0.057473,
        "ARIMA+SVR"     : 0.057498,
        "SVR(ARIMA,SVR)": 0.057903,
        "Proposto"      : 0.057498,
    },
    "HNEI" : {
        "ARIMA"         : 0.077471,
        "MLP"           : 0.094646,
        "SVR"           : 0.097411,
        "DTR"           : 0.158851,
        "ARIMA+MLP"     : 0.077303,
        "ARIMA+SVR"     : 0.077317,
        "SVR(ARIMA,SVR)": 0.077991,
        "Proposto"      : 0.077303,
    },
}


# ---------------------------------------------
# FUNCAO: Monta DataFrame de MSE/MAE
# Linhas = datasets | Colunas = modelos
# ---------------------------------------------
def build_dataframe(data_dict):
    df = pd.DataFrame(data_dict).T   # datasets x modelos
    return df


# ---------------------------------------------
# FUNCAO: Calcula ranks medios
# Rank 1 = melhor (menor MSE/MAE)
# ---------------------------------------------
def compute_avg_ranks(df):
    # Rank por linha (dataset): menor valor = rank 1
    ranked = df.rank(axis=1, ascending=True)
    avg_ranks = ranked.mean(axis=0).sort_values()

    print("\nRanks medios por modelo (1 = melhor):")
    for model, rank in avg_ranks.items():
        bar = "#" * int(rank * 5)
        print("  {:<20} {:.3f}  {}".format(model, rank, bar))

    return ranked, avg_ranks


# ---------------------------------------------
# FUNCAO: Teste de Friedman
# Verifica se ha diferenca global entre modelos
# ---------------------------------------------
def run_friedman(df):
    # Formato: lista de arrays, um por modelo
    groups = [df[col].values for col in df.columns]
    stat, p = friedmanchisquare(*groups)

    print("\nTeste de Friedman:")
    print("  Estatistica chi2 = {:.4f}".format(stat))
    print("  p-valor          = {:.6f}".format(p))

    if p < 0.05:
        print("  Resultado: Diferenca SIGNIFICATIVA (p < 0.05)")
        print("  -> Prossegue com teste post-hoc de Nemenyi")
    else:
        print("  Resultado: Sem diferenca significativa (p >= 0.05)")
        print("  -> Nemenyi aplicado de forma ilustrativa")
        print("  NOTA: Com dados sinteticos e apenas 5 datasets,")
        print("        o poder do teste e reduzido. Com dados reais")
        print("        e mais datasets o resultado seria diferente.")
    return stat, p


# ---------------------------------------------
# FUNCAO: Teste de Nemenyi
# ---------------------------------------------
def run_nemenyi(df):
    # scikit_posthocs espera: linhas = observacoes, colunas = grupos
    pval_matrix = sp.posthoc_nemenyi_friedman(df.values)
    pval_matrix.index   = df.columns
    pval_matrix.columns = df.columns

    print("\nMatriz de p-valores Nemenyi (alpha = 0.05):")
    print("  Valores < 0.05 indicam diferenca estatisticamente significativa")
    print(pval_matrix.round(3).to_string())

    path = os.path.join(RESULTS_PATH, "nemenyi_pvalues_MSE.csv")
    pval_matrix.to_csv(path)
    print("\n  Matriz salva em: {}".format(path))
    return pval_matrix


# ---------------------------------------------
# FUNCAO: Diagrama de distancia critica
# Replica a Figura 4 do artigo
# ---------------------------------------------
def plot_critical_difference(avg_ranks, pval_matrix, metric="MSE", alpha=0.05):
    models = avg_ranks.index.tolist()
    ranks  = avg_ranks.values
    n_models   = len(models)
    n_datasets = 5

    # Distancia critica (formula de Nemenyi)
    # CD = q_alpha * sqrt(k(k+1) / (6*N))
    # q_alpha para alpha=0.05 e k=8 modelos ~ 3.031 (tabela de Demsar)
    q_alpha = 3.031
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

    print("\nDistancia critica (CD) = {:.4f}".format(cd))
    print("  (q_alpha={} para k={} modelos, N={} datasets)".format(
        q_alpha, n_models, n_datasets))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Eixo de ranks
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_ylim(-1, n_models + 1)
    ax.invert_xaxis()
    ax.axhline(y=n_models, color="black", linewidth=0.5)

    # Marca a distancia critica no topo
    ax.annotate("", xy=(1.0, n_models + 0.5),
                xytext=(1.0 + cd, n_models + 0.5),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1.0 + cd / 2, n_models + 0.75, "CD={:.2f}".format(cd),
            ha="center", va="bottom", fontsize=9)

    # Plota cada modelo como ponto no eixo
    colors_map = {
        "Proposto"      : "crimson",
        "ARIMA+MLP"     : "steelblue",
        "ARIMA+SVR"     : "steelblue",
        "SVR(ARIMA,SVR)": "steelblue",
        "ARIMA"         : "gray",
        "MLP"           : "gray",
        "SVR"           : "gray",
        "DTR"           : "gray",
    }

    for i, (model, rank) in enumerate(zip(models, ranks)):
        color = colors_map.get(model, "gray")
        ax.plot(rank, n_models, "o", color=color,
                markersize=10, zorder=5)

        # Linha vertical e label
        if i % 2 == 0:
            y_text = n_models - 1.2 - (i // 2) * 1.1
            va     = "top"
        else:
            y_text = n_models - 1.2 - (i // 2) * 1.1
            va     = "top"

        ax.plot([rank, rank], [n_models, y_text],
                color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(rank, y_text - 0.15, model,
                ha="center", va="top", fontsize=8, color=color,
                fontweight="bold" if model == "Proposto" else "normal")
        ax.text(rank, y_text - 0.75, "({:.2f})".format(rank),
                ha="center", va="top", fontsize=7, color="gray")

    # Conecta modelos sem diferenca significativa (p > alpha)
    connected_pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            m1, m2 = models[i], models[j]
            if m1 in pval_matrix.index and m2 in pval_matrix.columns:
                if pval_matrix.loc[m1, m2] > alpha:
                    connected_pairs.append((ranks[i], ranks[j]))

    for r1, r2 in connected_pairs:
        y_conn = n_models + 0.15
        ax.plot([r1, r2], [y_conn, y_conn],
                color="black", linewidth=3.0, solid_capstyle="round",
                alpha=0.5)

    ax.set_xlabel("Rank medio ({})".format(metric), fontsize=10)
    ax.set_title(
        "[DADOS SINTETICOS] Diagrama de Distancia Critica — Nemenyi (alpha={})\n"
        "Modelos conectados por linha nao diferem significativamente".format(alpha),
        fontsize=10
    )
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Legenda
    patch_prop  = mpatches.Patch(color="crimson",    label="Modelo proposto")
    patch_hyb   = mpatches.Patch(color="steelblue",  label="Modelos hibridos")
    patch_base  = mpatches.Patch(color="gray",        label="Baselines simples")
    ax.legend(handles=[patch_prop, patch_hyb, patch_base],
              loc="lower left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_PATH,
                        "FIGURA4_nemenyi_{}.png".format(metric))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura 4 salva em: {}".format(path))
    return cd


# ---------------------------------------------
# FUNCAO: Heatmap da matriz de p-valores
# Material suplementar para o relatorio
# ---------------------------------------------
def plot_pvalue_heatmap(pval_matrix, metric="MSE"):
    fig, ax = plt.subplots(figsize=(9, 7))
    models  = pval_matrix.columns.tolist()
    n       = len(models)
    data    = pval_matrix.values.astype(float)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="p-valor")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)

    for i in range(n):
        for j in range(n):
            val   = data[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, "{:.3f}".format(val),
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        "[SINTETICO] Heatmap p-valores Nemenyi ({}) — verde=similar, vermelho=diferente".format(
            metric), fontsize=9)
    plt.tight_layout()

    path = os.path.join(RESULTS_PATH,
                        "nemenyi_heatmap_{}.png".format(metric))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Heatmap salvo em: {}".format(path))


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    print("=" * 55)
    print("  ETAPA 7 - TESTE DE NEMENYI")
    print("  Figura 4 do artigo | Secao 5")
    print("  Referencia: Demsar [35], alpha = 0.05")
    print("=" * 55)

    for metric, data_dict in [("MSE", MSE_DATA), ("MAE", MAE_DATA)]:

        print("\n" + "=" * 55)
        print("  Metrica: {}".format(metric))
        print("=" * 55)

        # Monta DataFrame
        df = build_dataframe(data_dict)
        print("\nDados utilizados ({}):\n".format(metric))
        print(df.to_string())

        # Ranks medios
        ranked, avg_ranks = compute_avg_ranks(df)

        # Teste de Friedman
        stat, p = run_friedman(df)

        # Teste de Nemenyi
        pval_matrix = run_nemenyi(df)

        # Diagrama de distancia critica (Figura 4)
        cd = plot_critical_difference(avg_ranks, pval_matrix, metric=metric)

        # Heatmap dos p-valores
        plot_pvalue_heatmap(pval_matrix, metric=metric)

        # Salva ranks
        rank_df = pd.DataFrame({
            "Modelo"    : avg_ranks.index,
            "Rank medio": avg_ranks.values.round(3),
        })
        path = os.path.join(RESULTS_PATH,
                            "nemenyi_ranks_{}.csv".format(metric))
        rank_df.to_csv(path, index=False)
        print("\n  Ranks salvos em: {}".format(path))

    print("\n" + "=" * 55)
    print("  RESUMO PARA O RELATORIO:")
    print("=" * 55)
    print("  Arquivos gerados:")
    print("  - FIGURA4_nemenyi_MSE.png  <- Figura 4 do artigo (MSE)")
    print("  - FIGURA4_nemenyi_MAE.png  <- Figura 4 do artigo (MAE)")
    print("  - nemenyi_heatmap_MSE.png  <- Heatmap p-valores")
    print("  - nemenyi_pvalues_MSE.csv  <- Matriz completa de p-valores")
    print("  - nemenyi_ranks_MSE.csv    <- Ranks medios por modelo")
    print("\n  NOTA SOBRE DADOS SINTETICOS:")
    print("  Com apenas 5 datasets o poder estatistico do teste")
    print("  e limitado. Resultados com dados reais seriam mais")
    print("  conclusivos, como demonstrado no artigo original.")
    print("=" * 55)