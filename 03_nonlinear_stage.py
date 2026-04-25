# 03_nonlinear_stage.py
# Estagio Nao-Linear - Treinamento do pool de modelos e selecao do melhor NM
# Corresponde a Secao 3.2 e Equacao 3 do artigo
# Pool: MLP, DTR, SVR (conforme Tabela 3 do artigo)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

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

WINDOW_SIZE = 5   # tamanho da janela k (Eq. 3 do artigo)


# ---------------------------------------------
# ESPACOS DE BUSCA (Tabela 3 do artigo)
# Versao reduzida para execucao mais rapida
# Os hiperparametros principais estao cobertos
# ---------------------------------------------
MLP_GRID = {
    "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100, 100)],
    "activation"        : ["relu", "tanh"],
    "solver"            : ["adam"],
    "alpha"             : [0.001, 0.01],
    "max_iter"          : [500],
    "random_state"      : [42],
}

DTR_GRID = {
    "splitter"        : ["best", "random"],
    "max_depth"       : [3, 5, 7, 9],
    "min_samples_leaf": [1, 3, 5],
    "max_features"    : ["sqrt", "log2", None],
    "random_state"    : [42],
}

SVR_GRID = {
    "kernel" : ["rbf"],
    "C"      : [0.1, 1, 10, 100],
    "epsilon": [0.1, 0.01],
    "gamma"  : [0.1, 0.01, 0.001],
}


# ---------------------------------------------
# FUNCOES DA ETAPA 1 (reaproveitadas)
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
    col_map  = {c.lower(): c for c in df.columns}
    real_col = col_map[target_col.lower()]
    raw      = df[real_col].dropna().values.astype(float)
    norm, z_min, z_max = minmax_normalize(raw)
    train, val, test   = split_dataset(norm)
    return {"name": dataset_name, "train": train, "val": val, "test": test}


# ---------------------------------------------
# FUNCOES DO ARIMA (reaproveitadas da Etapa 2)
# ---------------------------------------------
import pmdarima as pm

def fit_arima(train):
    model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    fitted          = model.predict_in_sample()
    train_aligned   = train[-len(fitted):]
    train_residuals = train_aligned - fitted
    return model, train_aligned, train_residuals

def arima_predict_seq(model, observations):
    preds = []
    for obs in observations:
        pred = model.predict(n_periods=1)[0]
        preds.append(pred)
        model.update([obs])
    preds     = np.array(preds)
    residuals = observations - preds
    return preds, residuals


# ---------------------------------------------
# FUNCAO: Janela deslizante -> (X, y)
# Eq. 3: N_t = f(e_{t-1}, ..., e_{t-k})
# ---------------------------------------------
def create_windows(series, window_size):
    """
    Transforma serie de residuos em pares (X, y).
    Cada linha de X contem k residuos anteriores.
    y e o residuo a ser previsto.
    window_size = k (tamanho da janela temporal)
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# ---------------------------------------------
# FUNCAO: Grid search para um modelo
# ---------------------------------------------
def grid_search(estimator, param_grid, X_train, y_train, model_name):
    print("    Grid search: {} ...".format(model_name), end=" ", flush=True)
    gs = GridSearchCV(
        estimator, param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print("OK | Melhor MSE CV: {:.6f}".format(-gs.best_score_))
    return gs.best_estimator_, gs.best_params_


# ---------------------------------------------
# FUNCAO: Treina pool e seleciona melhor NM
# 1o nivel de selecao do artigo (Secao 3.2)
# ---------------------------------------------
def train_and_select_nm(train_residuals, val_residuals, dataset_name):
    """
    Treina MLP, DTR e SVR nos residuos de treino.
    Seleciona o melhor avaliando no conjunto de VALIDACAO.
    Este e o 1o nivel de selecao proposto no artigo.
    """
    print("\n  [Pool Nao-Linear] Treinando modelos ...")

    # Janelas de treino
    X_tr, y_tr = create_windows(train_residuals, WINDOW_SIZE)

    # Janelas de validacao:
    # concatena treino + validacao para gerar contexto historico correto
    combined_val = np.concatenate([train_residuals, val_residuals])
    X_val_all, y_val_all = create_windows(combined_val, WINDOW_SIZE)
    X_val = X_val_all[-len(val_residuals):]
    y_val = y_val_all[-len(val_residuals):]

    # Treina os tres modelos com grid search
    best_params_log = {}

    mlp, mlp_params = grid_search(
        MLPRegressor(), MLP_GRID, X_tr, y_tr, "MLP")
    dtr, dtr_params = grid_search(
        DecisionTreeRegressor(), DTR_GRID, X_tr, y_tr, "DTR")
    svr, svr_params = grid_search(
        SVR(), SVR_GRID, X_tr, y_tr, "SVR")

    best_params_log = {
        "MLP": mlp_params,
        "DTR": dtr_params,
        "SVR": svr_params,
    }

    models = {"MLP": mlp, "DTR": dtr, "SVR": svr}

    # Avalia cada modelo no conjunto de VALIDACAO
    print("\n  [Selecao NM] Avaliando no conjunto de validacao:")
    val_scores = {}
    val_preds  = {}

    for nm_name, model in models.items():
        preds              = model.predict(X_val)
        mse                = mean_squared_error(y_val, preds)
        val_scores[nm_name] = mse
        val_preds[nm_name]  = preds
        print("    {} | MSE val: {:.6f}".format(nm_name, mse))

    # Seleciona o melhor (menor MSE na validacao)
    best_nm_name = min(val_scores, key=val_scores.get)
    best_nm      = models[best_nm_name]
    print("\n  >>> Melhor NM selecionado: {} (MSE val = {:.6f})".format(
        best_nm_name, val_scores[best_nm_name]))

    return {
        "models"         : models,
        "best_nm_name"   : best_nm_name,
        "best_nm"        : best_nm,
        "val_scores"     : val_scores,
        "val_preds"      : val_preds,
        "X_val"          : X_val,
        "y_val"          : y_val,
        "X_tr"           : X_tr,
        "y_tr"           : y_tr,
        "best_params_log": best_params_log,
    }


# ---------------------------------------------
# FUNCAO: Previsao NM no conjunto de teste
# ---------------------------------------------
def nm_predict_test(best_nm, train_residuals, val_residuals, test_residuals):
    combined_test = np.concatenate([train_residuals, val_residuals, test_residuals])
    X_test_all, y_test_all = create_windows(combined_test, WINDOW_SIZE)
    X_test = X_test_all[-len(test_residuals):]
    y_test = y_test_all[-len(test_residuals):]

    test_nm_preds = best_nm.predict(X_test)
    mse = mean_squared_error(y_test, test_nm_preds)
    mae = mean_absolute_error(y_test, test_nm_preds)
    print("  MSE teste (NM residuo): {:.6f}".format(mse))
    print("  MAE teste (NM residuo): {:.6f}".format(mae))

    return test_nm_preds, y_test, X_test


# ---------------------------------------------
# FUNCAO: Grafico comparativo dos modelos NM
# Para o relatorio: justifica a selecao adaptativa
# ---------------------------------------------
def plot_nm_comparison(val_scores, dataset_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = list(val_scores.keys())
    values = [val_scores[n] for n in names]
    colors = ["steelblue" if n != min(val_scores, key=val_scores.get)
              else "crimson" for n in names]

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("[SINTETICO] {} - MSE por modelo NM (validacao)".format(dataset_name))
    ax.set_ylabel("MSE (validacao)")
    ax.set_xlabel("Modelo")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00005,
                "{:.5f}".format(val),
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "{}_nm_comparison.png".format(dataset_name))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# FUNCAO: Tabela resumo da selecao de NM
# ---------------------------------------------
def summarize_nm_selection(nm_selection_log):
    rows = []
    for ds_name, res in nm_selection_log.items():
        row = {"Dataset": ds_name, "NM Selecionado": res["best_nm_name"]}
        for nm_name, mse in res["val_scores"].items():
            row["MSE val {}".format(nm_name)] = round(mse, 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nResumo da selecao do modelo nao-linear por dataset:")
    print(df.to_string(index=False))

    path = os.path.join(RESULTS_PATH, "nm_selection_summary.csv")
    df.to_csv(path, index=False)
    print("Tabela salva em: {}".format(path))
    return df


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    nm_selection_log = {}
    all_nm_results   = {}

    print("=" * 55)
    print("  ETAPA 3 - ESTAGIO NAO-LINEAR (SELECAO DO NM)")
    print("  Secao 3.2 | Equacao 3 do artigo")
    print("  Pool: MLP, DTR, SVR")
    print("=" * 55)
    print("  AVISO: Esta etapa pode demorar 10-20 minutos")
    print("         devido ao grid search em 3 modelos x 5 datasets.")
    print("=" * 55)

    for ds_name, ds_file in DATASETS.items():

        print("\n" + "-" * 55)
        print("  Dataset: {}".format(ds_name))
        print("-" * 55)

        # Carrega dados
        try:
            data = load_and_preprocess(ds_file, ds_name)
        except Exception as e:
            print("AVISO: Nao foi possivel carregar {}: {}".format(ds_name, e))
            continue

        train = data["train"]
        val   = data["val"]
        test  = data["test"]

        # Roda ARIMA para obter residuos
        print("  Rodando ARIMA para obter residuos ...")
        arima_model, train_aligned, train_residuals = fit_arima(train)

        print("  Obtendo residuos de validacao ...")
        val_arima_preds, val_residuals = arima_predict_seq(arima_model, val)

        print("  Obtendo residuos de teste ...")
        test_arima_preds, test_residuals = arima_predict_seq(arima_model, test)

        # Treina pool e seleciona melhor NM
        nm_res = train_and_select_nm(train_residuals, val_residuals, ds_name)

        # Previsao do melhor NM no teste
        print("\n  [Teste NM]")
        test_nm_preds, y_test_nm, X_test_nm = nm_predict_test(
            nm_res["best_nm"],
            train_residuals,
            val_residuals,
            test_residuals,
        )

        # Grafico comparativo dos modelos NM
        plot_nm_comparison(nm_res["val_scores"], ds_name)

        # Armazena resultados para Etapa 4
        nm_selection_log[ds_name] = {
            "best_nm_name": nm_res["best_nm_name"],
            "val_scores"  : nm_res["val_scores"],
        }

        all_nm_results[ds_name] = {
            "best_nm_name"    : nm_res["best_nm_name"],
            "best_nm"         : nm_res["best_nm"],
            "models"          : nm_res["models"],
            "train_residuals" : train_residuals,
            "val_residuals"   : val_residuals,
            "test_residuals"  : test_residuals,
            "val_arima_preds" : val_arima_preds,
            "test_arima_preds": test_arima_preds,
            "test_nm_preds"   : test_nm_preds,
            "train_aligned"   : train_aligned,
            "val"             : val,
            "test"            : test,
            "X_val"           : nm_res["X_val"],
            "y_val"           : nm_res["y_val"],
        }

    # Tabela resumo
    summarize_nm_selection(nm_selection_log)

    print("\n" + "=" * 55)
    print("  Etapa 3 concluida!")
    print("  Arquivos gerados em: results/")
    print("  Proximo passo: 04_combination_stage.py")
    print("=" * 55)