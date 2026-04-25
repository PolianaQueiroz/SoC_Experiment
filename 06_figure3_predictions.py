# 06_figure3_predictions.py
# Visualizacao das previsoes finais — equivalente a Figura 3 do artigo
# Compara: Valores reais | ARIMA | Modelo proposto
# Para cada um dos 5 datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

import pmdarima as pm
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

WINDOW_SIZE = 5

# Modelos selecionados nas Etapas 3 e 4
SELECTED = {
    "CALCE" : {"NM": "DTR", "CM": "SVR"},
    "OX"    : {"NM": "MLP", "CM": "DTR"},
    "UL-PUR": {"NM": "SVR", "CM": "SVR"},
    "SNL"   : {"NM": "SVR", "CM": "Summation"},
    "HNEI"  : {"NM": "MLP", "CM": "Summation"},
}

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
# FUNCOES UTILITARIAS
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
    return {"name": dataset_name, "train": train, "val": val,
            "test": test, "z_min": z_min, "z_max": z_max}

def create_windows(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def grid_search(estimator, param_grid, X_train, y_train):
    gs = GridSearchCV(estimator, param_grid, cv=3,
                      scoring="neg_mean_squared_error",
                      n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def arima_predict_seq(model, observations):
    preds = []
    for obs in observations:
        pred = model.predict(n_periods=1)[0]
        preds.append(pred)
        model.update([obs])
    return np.array(preds), observations - np.array(preds)

def get_nm_estimator(nm_name):
    if nm_name == "MLP":
        return MLPRegressor(), MLP_GRID
    elif nm_name == "DTR":
        return DecisionTreeRegressor(), DTR_GRID
    else:
        return SVR(), SVR_GRID


# ---------------------------------------------
# FUNCAO PRINCIPAL: Reproduz pipeline completo
# e retorna previsoes do ARIMA e do modelo proposto
# ---------------------------------------------
def run_full_pipeline(ds_name, train, val, test):
    nm_name = SELECTED[ds_name]["NM"]
    cm_name = SELECTED[ds_name]["CM"]

    print("  Rodando ARIMA ...", end=" ", flush=True)
    arima_model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    print("OK p={} d={} q={}".format(*arima_model.order))

    fitted          = arima_model.predict_in_sample()
    train_residuals = train[-len(fitted):] - fitted

    val_arima,  val_residuals  = arima_predict_seq(arima_model, val)
    test_arima, test_residuals = arima_predict_seq(arima_model, test)

    # NM selecionado
    print("  Treinando NM ({}) ...".format(nm_name), end=" ", flush=True)
    X_tr_res, y_tr_res = create_windows(train_residuals, WINDOW_SIZE)
    estimator, grid    = get_nm_estimator(nm_name)
    best_nm            = grid_search(estimator, grid, X_tr_res, y_tr_res)
    print("OK")

    # Previsoes NM na validacao
    combined_val  = np.concatenate([train_residuals, val_residuals])
    X_val_all, _  = create_windows(combined_val, WINDOW_SIZE)
    X_val_nm      = X_val_all[-len(val_residuals):]
    val_nm_preds  = best_nm.predict(X_val_nm)

    # Previsoes NM no teste
    combined_test = np.concatenate([train_residuals, val_residuals, test_residuals])
    X_te_all, _   = create_windows(combined_test, WINDOW_SIZE)
    X_te_nm       = X_te_all[-len(test_residuals):]
    test_nm_preds = best_nm.predict(X_te_nm)

    # CM selecionado
    print("  Aplicando CM ({}) ...".format(cm_name), end=" ", flush=True)
    n_val = min(len(val_arima), len(val_nm_preds), len(val))
    val_arima_a   = val_arima[-n_val:]
    val_nm_a      = val_nm_preds[-n_val:]
    val_targets_a = val[-n_val:]

    if cm_name == "Summation":
        n_te    = min(len(test_arima), len(test_nm_preds))
        final   = test_arima[-n_te:] + test_nm_preds[-n_te:]
    else:
        X_comb_val = np.column_stack([val_arima_a, val_nm_a])
        n_fit      = int(len(val_targets_a) * 0.7)

        if cm_name == "SVR":
            cm_model = SVR(kernel="rbf", C=10, epsilon=0.01, gamma=0.1)
        elif cm_name == "DTR":
            cm_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            cm_model = MLPRegressor(hidden_layer_sizes=(50,100,50),
                                    activation="relu", solver="adam",
                                    max_iter=500, random_state=42)

        cm_model.fit(X_comb_val[:n_fit], val_targets_a[:n_fit])

        n_te      = min(len(test_arima), len(test_nm_preds))
        X_comb_te = np.column_stack([test_arima[-n_te:], test_nm_preds[-n_te:]])
        final     = cm_model.predict(X_comb_te)

    print("OK")

    # Alinha todos os vetores para o mesmo tamanho
    n = min(len(test), len(test_arima), len(final))
    return test[-n:], test_arima[-n:], final[-n:]


# ---------------------------------------------
# GRAFICO INDIVIDUAL por dataset
# Replica cada subgrafico da Figura 3 do artigo
# ---------------------------------------------
def plot_individual(test_targets, arima_preds, proposed_preds,
                    dataset_name, ax=None):
    save_individual = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    t = range(len(test_targets))

    ax.plot(t, test_targets,   color="black",  linewidth=1.4,
            label="Valores reais", zorder=4)
    ax.plot(t, arima_preds,    color="blue",   linewidth=1.0,
            linestyle="--", label="ARIMA", alpha=0.85, zorder=3)
    ax.plot(t, proposed_preds, color="crimson", linewidth=1.2,
            label="Modelo proposto", alpha=0.9, zorder=5)

    ax.set_title("[SINTETICO] {}".format(dataset_name), fontsize=10)
    ax.set_xlabel("Passo de tempo", fontsize=8)
    ax.set_ylabel("SoC normalizado", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    if save_individual:
        plt.tight_layout()
        path = os.path.join(RESULTS_PATH,
                            "{}_fig3_prediction.png".format(dataset_name))
        plt.savefig(path, dpi=150)
        plt.show()
        print("  Grafico individual salvo em: {}".format(path))


# ---------------------------------------------
# GRAFICO COMBINADO — replica exata da Figura 3
# Todos os 5 subgraficos em um unico painel
# ---------------------------------------------
def plot_figure3(all_predictions):
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    positions = [(0,0), (0,1), (1,0), (1,1), (2,0)]
    labels    = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    for i, (ds_name, preds) in enumerate(all_predictions.items()):
        row, col = positions[i]
        ax = fig.add_subplot(gs[row, col])

        test_targets   = preds["test"]
        arima_preds    = preds["arima"]
        proposed_preds = preds["proposed"]

        plot_individual(test_targets, arima_preds, proposed_preds,
                        ds_name, ax=ax)

        ax.set_title("{} {} predictions".format(labels[i], ds_name),
                     fontsize=10)

    # Remove subplot vazio (posicao 2,1)
    fig.add_subplot(gs[2, 1]).set_visible(False)

    fig.suptitle("[DADOS SINTETICOS] Figura 3 — Previsoes por dataset\n"
                 "Comparacao: Valores reais | ARIMA | Modelo proposto",
                 fontsize=12, y=1.01)

    path = os.path.join(RESULTS_PATH, "FIGURA3_predictions_all_datasets.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigura 3 completa salva em: {}".format(path))


# ---------------------------------------------
# TABELA DE METRICAS FINAIS para o relatorio
# ---------------------------------------------
def print_metrics_table(all_predictions):
    rows = []
    for ds_name, preds in all_predictions.items():
        test     = preds["test"]
        arima    = preds["arima"]
        proposed = preds["proposed"]

        rows.append({
            "Dataset"      : ds_name,
            "MSE ARIMA"    : round(mean_squared_error(test, arima), 6),
            "MSE Proposto" : round(mean_squared_error(test, proposed), 6),
            "MAE ARIMA"    : round(mean_absolute_error(test, arima), 6),
            "MAE Proposto" : round(mean_absolute_error(test, proposed), 6),
        })

    df = pd.DataFrame(rows)
    df["Ganho MSE (%)"] = ((df["MSE ARIMA"] - df["MSE Proposto"])
                           / df["MSE ARIMA"] * 100).round(2)
    df["Ganho MAE (%)"] = ((df["MAE ARIMA"] - df["MAE Proposto"])
                           / df["MAE ARIMA"] * 100).round(2)

    print("\n" + "=" * 70)
    print("  METRICAS FINAIS — ARIMA vs MODELO PROPOSTO")
    print("=" * 70)
    print(df.to_string(index=False))

    path = os.path.join(RESULTS_PATH, "FIGURA3_metricas_resumo.csv")
    df.to_csv(path, index=False)
    print("\nTabela salva em: {}".format(path))
    return df


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    all_predictions = {}

    print("=" * 55)
    print("  ETAPA 6 - FIGURA 3 DO ARTIGO")
    print("  Visualizacao das previsoes finais")
    print("  Datasets: CALCE, OX, UL-PUR, SNL, HNEI")
    print("=" * 55)
    print("  AVISO: Pipeline completo por dataset.")
    print("         Tempo estimado: 15-25 minutos.")
    print("=" * 55)

    for ds_name, ds_file in DATASETS.items():

        print("\n" + "-" * 55)
        print("  Dataset: {}".format(ds_name))
        print("-" * 55)

        try:
            data = load_and_preprocess(ds_file, ds_name)
        except Exception as e:
            print("AVISO: {}".format(e))
            continue

        train = data["train"]
        val   = data["val"]
        test  = data["test"]

        # Roda pipeline completo
        test_targets, arima_preds, proposed_preds = run_full_pipeline(
            ds_name, train, val, test
        )

        # Salva resultados
        all_predictions[ds_name] = {
            "test"    : test_targets,
            "arima"   : arima_preds,
            "proposed": proposed_preds,
        }

        # Grafico individual
        plot_individual(test_targets, arima_preds, proposed_preds, ds_name)

        mse_arima    = mean_squared_error(test_targets, arima_preds)
        mse_proposed = mean_squared_error(test_targets, proposed_preds)
        print("  MSE ARIMA    : {:.6f}".format(mse_arima))
        print("  MSE Proposto : {:.6f}".format(mse_proposed))
        ganho = (mse_arima - mse_proposed) / mse_arima * 100
        print("  Ganho        : {:.2f}%".format(ganho))

    # Figura 3 completa (todos os datasets em um painel)
    plot_figure3(all_predictions)

    # Tabela de metricas
    print_metrics_table(all_predictions)

    print("\n" + "=" * 55)
    print("  Etapa 6 concluida!")
    print("  Arquivos gerados:")
    print("  - FIGURA3_predictions_all_datasets.png")
    print("  - [dataset]_fig3_prediction.png (x5)")
    print("  - FIGURA3_metricas_resumo.csv")
    print("  Proximo passo: 07_nemenyi_test.py")
    print("=" * 55)