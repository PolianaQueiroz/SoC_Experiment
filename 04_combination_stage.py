# 04_combination_stage.py
# Estagio de Combinacao - Selecao da melhor funcao de combinacao (CM)
# e geracao da previsao final Zt_chapeu = CM(Lt_chapeu, Nt_chapeu)
# Corresponde a Secao 3.3 e Equacao 13 do artigo
# 2o nivel de selecao proposto no artigo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Modelos NM selecionados na Etapa 3
NM_SELECTED = {
    "CALCE" : "DTR",
    "OX"    : "MLP",
    "UL-PUR": "SVR",
    "SNL"   : "SVR",
    "HNEI"  : "MLP",
}

# Espacos de busca (mesmos da Etapa 3)
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
# FUNCOES REAPROVEITADAS DAS ETAPAS ANTERIORES
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

def create_windows(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def grid_search(estimator, param_grid, X_train, y_train, model_name):
    print("    Grid search: {} ...".format(model_name), end=" ", flush=True)
    gs = GridSearchCV(estimator, param_grid, cv=3,
                      scoring="neg_mean_squared_error",
                      n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    print("OK")
    return gs.best_estimator_

def get_nm_model(nm_name):
    if nm_name == "MLP":
        return MLPRegressor(), MLP_GRID
    elif nm_name == "DTR":
        return DecisionTreeRegressor(), DTR_GRID
    else:
        return SVR(), SVR_GRID

def train_nm(train_residuals, nm_name):
    X_tr, y_tr   = create_windows(train_residuals, WINDOW_SIZE)
    estimator, grid = get_nm_model(nm_name)
    best_nm = grid_search(estimator, grid, X_tr, y_tr, nm_name)
    return best_nm

def predict_nm(best_nm, base_residuals, target_residuals):
    combined      = np.concatenate([base_residuals, target_residuals])
    X_all, y_all  = create_windows(combined, WINDOW_SIZE)
    X_target      = X_all[-len(target_residuals):]
    y_target      = y_all[-len(target_residuals):]
    preds         = best_nm.predict(X_target)
    return preds, y_target


# ---------------------------------------------
# FUNCAO: Combinacao por soma simples
# Eq. 4 do artigo: Zt = Lt + Nt
# Esta e a combinacao usada por Zhang [13]
# ---------------------------------------------
def summation(linear_preds, nonlinear_preds):
    return linear_preds + nonlinear_preds


# ---------------------------------------------
# FUNCAO: Treina e seleciona melhor CM
# 2o nivel de selecao do artigo (Secao 3.3)
# Pool: Summation, SVR, DTR, MLP
# ---------------------------------------------
def train_and_select_cm(val_linear_preds, val_nm_preds,
                        val_targets, dataset_name):
    """
    Treina o pool de combinadores no conjunto de validacao.
    Seleciona o melhor CM com base no menor MSE na validacao.
    Este e o 2o nivel de selecao proposto no artigo.

    Eq. 13: Zt_chapeu = CM(Lt_chapeu, Nt_chapeu)
    """
    print("\n  [Pool Combinacao] Selecionando funcao de combinacao ...")

    # Alinha os vetores (podem ter tamanhos ligeiramente diferentes)
    n = min(len(val_linear_preds), len(val_nm_preds), len(val_targets))
    val_linear_preds = val_linear_preds[-n:]
    val_nm_preds     = val_nm_preds[-n:]
    val_targets      = val_targets[-n:]

    X_comb = np.column_stack([val_linear_preds, val_nm_preds])

    # Pool de combinadores
    combiners = {
        "Summation": None,
        "SVR"      : SVR(kernel="rbf", C=10, epsilon=0.01, gamma=0.1),
        "DTR"      : DecisionTreeRegressor(max_depth=5, random_state=42),
        "MLP"      : MLPRegressor(hidden_layer_sizes=(50, 100, 50),
                                  activation="relu", solver="adam",
                                  max_iter=500, random_state=42),
    }

    val_scores = {}
    val_preds  = {}
    n_fit      = int(len(val_targets) * 0.7)

    for cm_name, cm_model in combiners.items():

        if cm_name == "Summation":
            preds = summation(val_linear_preds, val_nm_preds)
        else:
            cm_model.fit(X_comb[:n_fit], val_targets[:n_fit])
            preds = cm_model.predict(X_comb)
            combiners[cm_name] = cm_model

        mse                = mean_squared_error(val_targets, preds)
        val_scores[cm_name] = mse
        val_preds[cm_name]  = preds
        print("    {} | MSE val: {:.6f}".format(cm_name, mse))

    best_cm_name = min(val_scores, key=val_scores.get)
    best_cm      = combiners[best_cm_name]
    print("\n  >>> Melhor CM selecionado: {} (MSE val = {:.6f})".format(
        best_cm_name, val_scores[best_cm_name]))

    return {
        "combiners"    : combiners,
        "best_cm_name" : best_cm_name,
        "best_cm"      : best_cm,
        "val_scores"   : val_scores,
        "val_preds"    : val_preds,
        "val_targets"  : val_targets,
    }


# ---------------------------------------------
# FUNCAO: Previsao final no teste
# Eq. 13: Zt_chapeu = CM(Lt_chapeu, Nt_chapeu)
# ---------------------------------------------
def final_predict(cm_res, test_linear_preds, test_nm_preds, test_targets):
    best_cm_name = cm_res["best_cm_name"]
    best_cm      = cm_res["best_cm"]

    n = min(len(test_linear_preds), len(test_nm_preds), len(test_targets))
    test_linear_preds = test_linear_preds[-n:]
    test_nm_preds     = test_nm_preds[-n:]
    test_targets      = test_targets[-n:]

    if best_cm_name == "Summation":
        final_preds = summation(test_linear_preds, test_nm_preds)
    else:
        X_test_comb = np.column_stack([test_linear_preds, test_nm_preds])
        final_preds = best_cm.predict(X_test_comb)

    mse = mean_squared_error(test_targets, final_preds)
    mae = mean_absolute_error(test_targets, final_preds)

    print("\n  [RESULTADO FINAL] CM: {}".format(best_cm_name))
    print("  MSE teste (modelo proposto): {:.6f}".format(mse))
    print("  MAE teste (modelo proposto): {:.6f}".format(mae))

    return final_preds, test_targets, mse, mae


# ---------------------------------------------
# FUNCAO: Grafico de previsoes finais
# Equivalente a Figura 3 do artigo
# ---------------------------------------------
def plot_final_predictions(test_targets, arima_preds,
                           final_preds, dataset_name):
    n = min(len(test_targets), len(arima_preds), len(final_preds))
    t = range(n)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, test_targets[-n:], color="black",   linewidth=1.4,
            label="Valores reais", zorder=3)
    ax.plot(t, arima_preds[-n:],  color="blue",    linewidth=1.0,
            linestyle="--", label="ARIMA", alpha=0.8)
    ax.plot(t, final_preds[-n:],  color="crimson", linewidth=1.2,
            label="Modelo proposto", alpha=0.9)

    ax.set_title("[SINTETICO] {} - Previsao final vs ARIMA (teste)".format(
        dataset_name))
    ax.set_xlabel("Passo de tempo")
    ax.set_ylabel("SoC normalizado")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_PATH,
                        "{}_final_predictions.png".format(dataset_name))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# FUNCAO: Grafico comparativo do pool de CM
# ---------------------------------------------
def plot_cm_comparison(val_scores, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    names   = list(val_scores.keys())
    values  = [val_scores[n] for n in names]
    best    = min(val_scores, key=val_scores.get)
    colors  = ["crimson" if n == best else "steelblue" for n in names]

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("[SINTETICO] {} - MSE por combinador (validacao)".format(
        dataset_name))
    ax.set_ylabel("MSE (validacao)")
    ax.set_xlabel("Combinador")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.000005,
                "{:.5f}".format(val),
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH,
                        "{}_cm_comparison.png".format(dataset_name))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    final_results  = {}
    selected_cms   = {}

    print("=" * 55)
    print("  ETAPA 4 - ESTAGIO DE COMBINACAO (SELECAO DO CM)")
    print("  Secao 3.3 | Equacao 13 do artigo")
    print("  Pool: Summation, SVR, DTR, MLP")
    print("=" * 55)
    print("  AVISO: Esta etapa pode demorar 10-20 minutos.")
    print("=" * 55)

    for ds_name, ds_file in DATASETS.items():

        print("\n" + "-" * 55)
        print("  Dataset: {}".format(ds_name))
        print("-" * 55)

        # Carrega dados
        try:
            data = load_and_preprocess(ds_file, ds_name)
        except Exception as e:
            print("AVISO: {}".format(e))
            continue

        train = data["train"]
        val   = data["val"]
        test  = data["test"]

        # ARIMA
        print("  Rodando ARIMA ...")
        arima_model, train_aligned, train_residuals = fit_arima(train)
        val_arima_preds,  val_residuals  = arima_predict_seq(arima_model, val)
        test_arima_preds, test_residuals = arima_predict_seq(arima_model, test)

        # Treina o NM selecionado na Etapa 3
        nm_name = NM_SELECTED[ds_name]
        print("  Treinando NM selecionado: {} ...".format(nm_name))
        best_nm = train_nm(train_residuals, nm_name)

        # Previsoes do NM na validacao e teste
        val_nm_preds,  _ = predict_nm(best_nm, train_residuals, val_residuals)
        test_nm_preds, _ = predict_nm(
            best_nm,
            np.concatenate([train_residuals, val_residuals]),
            test_residuals
        )

        # Alinha vetores de validacao
        n_val = min(len(val_arima_preds), len(val_nm_preds), len(val))
        val_arima_preds_a = val_arima_preds[-n_val:]
        val_nm_preds_a    = val_nm_preds[-n_val:]
        val_targets_a     = val[-n_val:]

        # Seleciona melhor CM (2o nivel de selecao)
        cm_res = train_and_select_cm(
            val_arima_preds_a,
            val_nm_preds_a,
            val_targets_a,
            ds_name,
        )

        # Previsao final no teste
        final_preds, test_targets, mse, mae = final_predict(
            cm_res,
            test_arima_preds,
            test_nm_preds,
            test,
        )

        # Graficos
        plot_cm_comparison(cm_res["val_scores"], ds_name)
        plot_final_predictions(test_targets, test_arima_preds,
                               final_preds, ds_name)

        # Armazena resultados
        selected_cms[ds_name] = {
            "NM": nm_name,
            "CM": cm_res["best_cm_name"],
        }
        final_results[ds_name] = {
            "MSE_proposto"   : round(mse, 6),
            "MAE_proposto"   : round(mae, 6),
            "MSE_ARIMA"      : round(float(mean_squared_error(
                                   test_targets, test_arima_preds[-len(test_targets):])), 6),
            "MAE_ARIMA"      : round(float(mean_absolute_error(
                                   test_targets, test_arima_preds[-len(test_targets):])), 6),
            "final_preds"    : final_preds,
            "test_targets"   : test_targets,
            "arima_preds"    : test_arima_preds,
        }

    # ------------------------------------------
    # TABELA FINAL DE RESULTADOS
    # Equivalente as Tabelas 5 e 6 do artigo
    # ------------------------------------------
    print("\n" + "=" * 55)
    print("  TABELA FINAL DE RESULTADOS")
    print("=" * 55)

    rows_mse, rows_mae = [], []
    for ds_name, res in final_results.items():
        rows_mse.append({
            "Dataset"        : ds_name,
            "MSE ARIMA"      : res["MSE_ARIMA"],
            "MSE Proposto"   : res["MSE_proposto"],
            "Ganho MSE (%)"  : round(
                (res["MSE_ARIMA"] - res["MSE_proposto"]) / res["MSE_ARIMA"] * 100, 2)
        })
        rows_mae.append({
            "Dataset"        : ds_name,
            "MAE ARIMA"      : res["MAE_ARIMA"],
            "MAE Proposto"   : res["MAE_proposto"],
            "Ganho MAE (%)"  : round(
                (res["MAE_ARIMA"] - res["MAE_proposto"]) / res["MAE_ARIMA"] * 100, 2)
        })

    df_mse = pd.DataFrame(rows_mse)
    df_mae = pd.DataFrame(rows_mae)

    print("\nResultados MSE (equivalente a Tabela 5 do artigo):")
    print(df_mse.to_string(index=False))
    df_mse.to_csv(os.path.join(RESULTS_PATH, "final_results_MSE.csv"), index=False)

    print("\nResultados MAE (equivalente a Tabela 6 do artigo):")
    print(df_mae.to_string(index=False))
    df_mae.to_csv(os.path.join(RESULTS_PATH, "final_results_MAE.csv"), index=False)

    # Tabela de modelos selecionados
    df_sel = pd.DataFrame([
        {"Dataset": k, "NM Selecionado": v["NM"], "CM Selecionado": v["CM"]}
        for k, v in selected_cms.items()
    ])
    print("\nModelos selecionados por dataset (equiv. Tabela 4 do artigo):")
    print(df_sel.to_string(index=False))
    df_sel.to_csv(os.path.join(RESULTS_PATH, "selected_models.csv"), index=False)

    print("\n" + "=" * 55)
    print("  Etapa 4 concluida!")
    print("  Arquivos gerados em: results/")
    print("  Proximo passo: 05_evaluation.py (Nemenyi)")
    print("=" * 55)