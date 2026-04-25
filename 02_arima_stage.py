# 02_arima_stage.py
# Estagio Linear - Ajuste do modelo ARIMA e calculo dos residuos
# Corresponde a Secao 3.1 e Equacoes 2 e 12 do artigo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import pmdarima as pm

warnings.filterwarnings("ignore")

# ---------------------------------------------
# CONFIGURACOES (mesmas da Etapa 1)
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
# FUNCOES DA ETAPA 1 (reaproveitadas aqui)
# ---------------------------------------------
def minmax_normalize(series):
    z_min = series.min()
    z_max = series.max()
    return (series - z_min) / (z_max - z_min), z_min, z_max


def split_dataset(series, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    n       = len(series)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))
    return series[:i_train], series[i_train:i_val], series[i_val:]


def load_and_preprocess(csv_filename, dataset_name, target_col="SoC"):
    filepath = os.path.join(DATA_PATH, csv_filename)
    df       = pd.read_csv(filepath)

    col_map = {c.lower(): c for c in df.columns}
    if target_col.lower() not in col_map:
        raise ValueError(
            "Coluna '{}' nao encontrada em {}. Colunas: {}".format(
                target_col, csv_filename, list(df.columns))
        )
    real_col   = col_map[target_col.lower()]
    raw_series = df[real_col].dropna().values.astype(float)

    norm_series, z_min, z_max = minmax_normalize(raw_series)
    train, val, test          = split_dataset(norm_series)

    return {
        "name": dataset_name, "raw": raw_series,
        "normalized": norm_series,
        "train": train, "val": val, "test": test,
        "z_min": z_min, "z_max": z_max,
    }


# ---------------------------------------------
# FUNCAO: Ajuste do ARIMA
# ---------------------------------------------
def fit_arima(train, dataset_name):
    """
    Ajusta ARIMA com selecao automatica via AIC.
    Equivalente ao auto.arima do R (Secao 4.3 do artigo).
    Espaco de busca: p in {1..5}, d in {1,2}, q in {1..5}
    """
    print("  Ajustando ARIMA ...", end=" ", flush=True)

    model = pm.auto_arima(
        train,
        start_p=1, max_p=5,
        start_q=1, max_q=5,
        d=None, max_d=2,
        seasonal=False,
        information_criterion='aic',
        stepwise=True,
        error_action='ignore',
        suppress_warnings=True,
        trace=False,
    )

    p, d, q = model.order
    print("OK  ->  p={}, d={}, q={}  |  AIC={:.4f}".format(p, d, q, model.aic()))

    train_fitted    = model.predict_in_sample()
    n_fitted        = len(train_fitted)
    train_aligned   = train[-n_fitted:]
    train_residuals = train_aligned - train_fitted

    print("  MSE treino (residuo)  : {:.6f}".format(np.mean(train_residuals**2)))

    return {
        "model"           : model,
        "order"           : (p, d, q),
        "train_fitted"    : train_fitted,
        "train_aligned"   : train_aligned,
        "train_residuals" : train_residuals,
    }


# ---------------------------------------------
# FUNCAO: Previsao na validacao
# ---------------------------------------------
def arima_predict_val(arima_result, val):
    """
    Previsao passo a passo na validacao (Eq. 2: Et = Zt - Lt).
    Atualiza o modelo a cada observacao real.
    """
    model     = arima_result["model"]
    val_preds = []

    for obs in val:
        pred = model.predict(n_periods=1)[0]
        val_preds.append(pred)
        model.update([obs])

    val_preds     = np.array(val_preds)
    val_residuals = val - val_preds

    print("  MSE validacao (ARIMA) : {:.6f}".format(np.mean(val_residuals**2)))
    return val_preds, val_residuals


# ---------------------------------------------
# FUNCAO: Previsao no teste
# ---------------------------------------------
def arima_predict_test(arima_result, test):
    """
    Previsao passo a passo no teste.
    O modelo ja foi atualizado com a validacao.
    """
    model      = arima_result["model"]
    test_preds = []

    for obs in test:
        pred = model.predict(n_periods=1)[0]
        test_preds.append(pred)
        model.update([obs])

    test_preds     = np.array(test_preds)
    test_residuals = test - test_preds

    print("  MSE teste  (ARIMA)    : {:.6f}".format(np.mean(test_residuals**2)))
    print("  MAE teste  (ARIMA)    : {:.6f}".format(np.mean(np.abs(test_residuals))))
    return test_preds, test_residuals


# ---------------------------------------------
# FUNCAO: Grafico dos residuos
# ---------------------------------------------
def plot_residuals(residuals, dataset_name, split="validacao"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(residuals, color="darkorange", linewidth=0.8)
    axes[0].axhline(0, linestyle="--", color="gray", linewidth=0.8)
    axes[0].set_title("[SINTETICO] {} - Residuos ARIMA ({})".format(dataset_name, split))
    axes[0].set_xlabel("Passo de tempo")
    axes[0].set_ylabel("Residuo (Et)")

    axes[1].hist(residuals, bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("Distribuicao dos residuos - {}".format(dataset_name))
    axes[1].set_xlabel("Residuo")
    axes[1].set_ylabel("Frequencia")

    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "{}_residuals_{}.png".format(dataset_name, split))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# FUNCAO: Grafico das previsoes ARIMA
# ---------------------------------------------
def plot_arima_predictions(real, preds, dataset_name, split="teste"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(real,  color="black", linewidth=1.2, label="Valores reais")
    ax.plot(preds, color="blue",  linewidth=1.0,
            linestyle="--", label="ARIMA", alpha=0.85)
    ax.set_title("[SINTETICO] {} - Previsoes ARIMA ({})".format(dataset_name, split))
    ax.set_xlabel("Passo de tempo")
    ax.set_ylabel("SoC normalizado")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_PATH, "{}_arima_{}.png".format(dataset_name, split))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# FUNCAO: Tabela resumo dos parametros ARIMA
# ---------------------------------------------
def summarize_arima_params(arima_results):
    rows = []
    for name, res in arima_results.items():
        p, d, q = res["order"]
        rows.append({
            "Dataset"         : name,
            "p"               : p,
            "d"               : d,
            "q"               : q,
            "AIC (treino)"    : round(res["model"].aic(), 4),
            "MSE resid treino": round(float(np.mean(res["train_residuals"]**2)), 6),
        })

    df = pd.DataFrame(rows)
    print("\nParametros ARIMA selecionados por dataset (equiv. Tabela 3 do artigo):")
    print(df.to_string(index=False))

    path = os.path.join(RESULTS_PATH, "arima_parameters.csv")
    df.to_csv(path, index=False)
    print("\nTabela salva em: {}".format(path))
    return df


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    all_arima_results = {}

    print("=" * 55)
    print("  ETAPA 2 - ESTAGIO LINEAR (ARIMA)")
    print("  Secao 3.1 | Equacoes 2 e 12 do artigo")
    print("=" * 55)

    for ds_name, ds_file in DATASETS.items():

        print("\n" + "-" * 55)
        print("  Dataset: {}".format(ds_name))
        print("-" * 55)

        try:
            data = load_and_preprocess(ds_file, ds_name, target_col="SoC")
        except Exception as e:
            print("AVISO: Nao foi possivel carregar {}: {}".format(ds_name, e))
            continue

        train = data["train"]
        val   = data["val"]
        test  = data["test"]

        # 1) Ajusta ARIMA no treino
        arima_res = fit_arima(train, ds_name)

        # 2) Previsao na validacao
        print("\n  [Validacao]")
        val_preds, val_residuals = arima_predict_val(arima_res, val)

        # 3) Previsao no teste
        print("\n  [Teste]")
        test_preds, test_residuals = arima_predict_test(arima_res, test)

        # 4) Graficos
        plot_residuals(val_residuals, ds_name, split="validacao")
        plot_arima_predictions(test, test_preds, ds_name, split="teste")

        # 5) Armazena tudo para as proximas etapas
        all_arima_results[ds_name] = {
            "model"           : arima_res["model"],
            "order"           : arima_res["order"],
            "train_fitted"    : arima_res["train_fitted"],
            "train_aligned"   : arima_res["train_aligned"],
            "train_residuals" : arima_res["train_residuals"],
            "val_preds"       : val_preds,
            "val_residuals"   : val_residuals,
            "test_preds"      : test_preds,
            "test_residuals"  : test_residuals,
            "train"           : train,
            "val"             : val,
            "test"            : test,
        }

    # Tabela final de parametros
    summarize_arima_params(all_arima_results)

    print("\n" + "=" * 55)
    print("  Etapa 2 concluida!")
    print("  Arquivos gerados em: results/")
    print("  Proximo passo: 03_nonlinear_stage.py")
    print("=" * 55)