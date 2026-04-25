# 05_baselines.py
# Treinamento de todos os modelos baseline para comparacao
# Corresponde as Tabelas 5, 6 e 7 do artigo
# Baselines: ARIMA, MLP, SVR, DTR, ARIMA+MLP, ARIMA+SVR, SVR(ARIMA,SVR)

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

# Resultados do modelo proposto (Etapa 4)
PROPOSED_RESULTS = {
    "CALCE" : {"MSE": 0.007573, "MAE": 0.068155},
    "OX"    : {"MSE": 0.017368, "MAE": 0.102772},
    "UL-PUR": {"MSE": 0.008249, "MAE": 0.072689},
    "SNL"   : {"MSE": 0.005183, "MAE": 0.057498},
    "HNEI"  : {"MSE": 0.009541, "MAE": 0.077303},
}

# Espacos de busca reduzidos (Tabela 3 do artigo)
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
    return {"name": dataset_name, "train": train, "val": val, "test": test}

def create_windows(series, window_size):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def grid_search(estimator, param_grid, X_train, y_train, name):
    print("    Grid search: {} ...".format(name), end=" ", flush=True)
    gs = GridSearchCV(estimator, param_grid, cv=3,
                      scoring="neg_mean_squared_error",
                      n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    print("OK")
    return gs.best_estimator_

def arima_predict_seq(model, observations):
    preds = []
    for obs in observations:
        pred = model.predict(n_periods=1)[0]
        preds.append(pred)
        model.update([obs])
    return np.array(preds), observations - np.array(preds)

def calc_metrics(y_true, y_pred):
    n   = min(len(y_true), len(y_pred))
    mse = mean_squared_error(y_true[-n:], y_pred[-n:])
    mae = mean_absolute_error(y_true[-n:], y_pred[-n:])
    return round(mse, 6), round(mae, 6)


# ---------------------------------------------
# BASELINE 1: ARIMA isolado
# Referencia [31] do artigo
# ---------------------------------------------
def run_arima(train, val, test):
    print("  [ARIMA] Ajustando ...", end=" ", flush=True)
    model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    print("OK p={} d={} q={}".format(*model.order))
    _, _ = arima_predict_seq(model, val)      # atualiza com val
    preds, _ = arima_predict_seq(model, test) # prevê teste
    mse, mae = calc_metrics(test, preds)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return preds, mse, mae, model


# ---------------------------------------------
# BASELINE 2: MLP isolado
# Referencia [32] do artigo
# ---------------------------------------------
def run_mlp(train, val, test):
    print("  [MLP standalone]")
    X_tr, y_tr = create_windows(train, WINDOW_SIZE)
    X_te_full, y_te_full = create_windows(
        np.concatenate([train, val, test]), WINDOW_SIZE)
    X_te = X_te_full[-len(test):]
    y_te = y_te_full[-len(test):]

    mlp = grid_search(MLPRegressor(), MLP_GRID, X_tr, y_tr, "MLP")
    preds = mlp.predict(X_te)
    mse, mae = calc_metrics(y_te, preds)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return preds, mse, mae


# ---------------------------------------------
# BASELINE 3: SVR isolado
# Referencia [33] do artigo
# ---------------------------------------------
def run_svr(train, val, test):
    print("  [SVR standalone]")
    X_tr, y_tr = create_windows(train, WINDOW_SIZE)
    X_te_full, y_te_full = create_windows(
        np.concatenate([train, val, test]), WINDOW_SIZE)
    X_te = X_te_full[-len(test):]
    y_te = y_te_full[-len(test):]

    svr = grid_search(SVR(), SVR_GRID, X_tr, y_tr, "SVR")
    preds = svr.predict(X_te)
    mse, mae = calc_metrics(y_te, preds)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return preds, mse, mae


# ---------------------------------------------
# BASELINE 4: DTR isolado
# Referencia [34] do artigo
# ---------------------------------------------
def run_dtr(train, val, test):
    print("  [DTR standalone]")
    X_tr, y_tr = create_windows(train, WINDOW_SIZE)
    X_te_full, y_te_full = create_windows(
        np.concatenate([train, val, test]), WINDOW_SIZE)
    X_te = X_te_full[-len(test):]
    y_te = y_te_full[-len(test):]

    dtr = grid_search(DecisionTreeRegressor(), DTR_GRID, X_tr, y_tr, "DTR")
    preds = dtr.predict(X_te)
    mse, mae = calc_metrics(y_te, preds)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return preds, mse, mae


# ---------------------------------------------
# BASELINE 5: ARIMA + MLP (Zhang [13])
# Combinacao por soma simples: Zt = Lt + Nt
# ---------------------------------------------
def run_arima_mlp(train, val, test):
    print("  [ARIMA+MLP]")
    # ARIMA
    model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    fitted          = model.predict_in_sample()
    train_residuals = train[-len(fitted):] - fitted

    val_preds, val_residuals   = arima_predict_seq(model, val)
    test_preds, test_residuals = arima_predict_seq(model, test)

    # MLP nos residuos
    X_tr, y_tr = create_windows(train_residuals, WINDOW_SIZE)
    combined   = np.concatenate([train_residuals, val_residuals, test_residuals])
    X_all, _   = create_windows(combined, WINDOW_SIZE)
    X_te       = X_all[-len(test_residuals):]

    mlp = grid_search(MLPRegressor(), MLP_GRID, X_tr, y_tr, "MLP")
    nm_preds = mlp.predict(X_te)

    # Combinacao por soma
    n     = min(len(test_preds), len(nm_preds))
    final = test_preds[-n:] + nm_preds[-n:]
    mse, mae = calc_metrics(test[-n:], final)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return final, mse, mae


# ---------------------------------------------
# BASELINE 6: ARIMA + SVR (Pai e Lin [18])
# Combinacao por soma simples
# ---------------------------------------------
def run_arima_svr(train, val, test):
    print("  [ARIMA+SVR]")
    model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    fitted          = model.predict_in_sample()
    train_residuals = train[-len(fitted):] - fitted

    val_preds, val_residuals   = arima_predict_seq(model, val)
    test_preds, test_residuals = arima_predict_seq(model, test)

    # SVR nos residuos
    X_tr, y_tr = create_windows(train_residuals, WINDOW_SIZE)
    combined   = np.concatenate([train_residuals, val_residuals, test_residuals])
    X_all, _   = create_windows(combined, WINDOW_SIZE)
    X_te       = X_all[-len(test_residuals):]

    svr = grid_search(SVR(), SVR_GRID, X_tr, y_tr, "SVR")
    nm_preds = svr.predict(X_te)

    n     = min(len(test_preds), len(nm_preds))
    final = test_preds[-n:] + nm_preds[-n:]
    mse, mae = calc_metrics(test[-n:], final)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return final, mse, mae


# ---------------------------------------------
# BASELINE 7: SVR(ARIMA, SVR) (Lucena [21])
# SVR como funcao de combinacao nao-linear
# ---------------------------------------------
def run_svr_arima_svr(train, val, test):
    print("  [SVR(ARIMA,SVR)]")
    model = pm.auto_arima(
        train, start_p=1, max_p=5, start_q=1, max_q=5,
        d=None, max_d=2, seasonal=False,
        information_criterion='aic', stepwise=True,
        error_action='ignore', suppress_warnings=True, trace=False,
    )
    fitted          = model.predict_in_sample()
    train_residuals = train[-len(fitted):] - fitted
    train_aligned   = train[-len(fitted):]

    val_arima,  val_residuals  = arima_predict_seq(model, val)
    test_arima, test_residuals = arima_predict_seq(model, test)

    # SVR nos residuos
    X_tr_res, y_tr_res = create_windows(train_residuals, WINDOW_SIZE)
    combined_res       = np.concatenate([train_residuals, val_residuals, test_residuals])
    X_all_res, _       = create_windows(combined_res, WINDOW_SIZE)
    X_val_res = X_all_res[-(len(val_residuals) + len(test_residuals)):-len(test_residuals)]
    X_te_res  = X_all_res[-len(test_residuals):]

    svr_nm = grid_search(SVR(), SVR_GRID, X_tr_res, y_tr_res, "SVR (NM)")
    val_nm_preds  = svr_nm.predict(X_val_res)
    test_nm_preds = svr_nm.predict(X_te_res)

    # SVR como combinador — treinado na validacao
    n_val  = min(len(val_arima), len(val_nm_preds), len(val))
    X_comb_val = np.column_stack([val_arima[-n_val:], val_nm_preds[-n_val:]])
    y_comb_val = val[-n_val:]

    svr_cm = SVR(kernel="rbf", C=10, epsilon=0.01, gamma=0.1)
    svr_cm.fit(X_comb_val, y_comb_val)

    n_te   = min(len(test_arima), len(test_nm_preds))
    X_comb_te = np.column_stack([test_arima[-n_te:], test_nm_preds[-n_te:]])
    final     = svr_cm.predict(X_comb_te)

    mse, mae = calc_metrics(test[-n_te:], final)
    print("  MSE: {}  MAE: {}".format(mse, mae))
    return final, mse, mae


# ---------------------------------------------
# GRAFICO: Comparacao de todos os modelos
# Equivalente a Figura 3 do artigo
# ---------------------------------------------
def plot_all_models(test_targets, predictions_dict, dataset_name):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Filtra apenas previsoes com dados (ignora listas vazias)
    valid_preds = {k: v for k, v in predictions_dict.items() if len(v) > 0}

    t = range(min(len(test_targets),
                  min(len(p) for p in valid_preds.values())))

    colors = {
        "Valores reais"   : ("black",    1.5, "-"),
        "ARIMA"           : ("blue",     0.9, "--"),
        "MLP"             : ("green",    0.9, "--"),
        "SVR"             : ("orange",   0.9, "--"),
        "DTR"             : ("purple",   0.9, "--"),
        "ARIMA+MLP"       : ("cyan",     0.9, "-."),
        "ARIMA+SVR"       : ("magenta",  0.9, "-."),
        "SVR(ARIMA,SVR)"  : ("brown",    0.9, "-."),
        "Proposto"        : ("crimson",  1.4, "-"),
    }

    ax.plot(t, test_targets[:len(t)], color="black", linewidth=1.5,
            label="Valores reais", zorder=5)

    for name, preds in valid_preds.items():
        c, lw, ls = colors.get(name, ("gray", 0.9, "--"))
        ax.plot(t, np.array(preds)[-len(t):],
                color=c, linewidth=lw, linestyle=ls,
                label=name, alpha=0.85)

    ax.set_title("[SINTETICO] {} - Comparacao de todos os modelos (teste)".format(
        dataset_name))
    ax.set_xlabel("Passo de tempo")
    ax.set_ylabel("SoC normalizado")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()

    path = os.path.join(RESULTS_PATH,
                        "{}_all_models_comparison.png".format(dataset_name))
    plt.savefig(path, dpi=150)
    plt.show()
    print("  Grafico salvo em: {}".format(path))


# ---------------------------------------------
# EXECUCAO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":

    all_results = {}

    print("=" * 55)
    print("  ETAPA 5 - MODELOS BASELINE")
    print("  Tabelas 5, 6 e 7 do artigo")
    print("  Baselines: ARIMA, MLP, SVR, DTR,")
    print("             ARIMA+MLP, ARIMA+SVR,")
    print("             SVR(ARIMA,SVR)")
    print("=" * 55)
    print("  AVISO: Esta etapa pode demorar 30-50 minutos.")
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

        preds_dict = {}
        results    = {}

        # 1) ARIMA
        arima_preds, mse, mae, _ = run_arima(train, val, test)
        results["ARIMA"]         = {"MSE": mse, "MAE": mae}
        preds_dict["ARIMA"]      = arima_preds

        # 2) MLP
        mlp_preds, mse, mae = run_mlp(train, val, test)
        results["MLP"]       = {"MSE": mse, "MAE": mae}
        preds_dict["MLP"]    = mlp_preds

        # 3) SVR
        svr_preds, mse, mae = run_svr(train, val, test)
        results["SVR"]       = {"MSE": mse, "MAE": mae}
        preds_dict["SVR"]    = svr_preds

        # 4) DTR
        dtr_preds, mse, mae = run_dtr(train, val, test)
        results["DTR"]       = {"MSE": mse, "MAE": mae}
        preds_dict["DTR"]    = dtr_preds

        # 5) ARIMA+MLP
        amp_preds, mse, mae   = run_arima_mlp(train, val, test)
        results["ARIMA+MLP"]  = {"MSE": mse, "MAE": mae}
        preds_dict["ARIMA+MLP"] = amp_preds

        # 6) ARIMA+SVR
        asv_preds, mse, mae   = run_arima_svr(train, val, test)
        results["ARIMA+SVR"]  = {"MSE": mse, "MAE": mae}
        preds_dict["ARIMA+SVR"] = asv_preds

        # 7) SVR(ARIMA,SVR)
        sas_preds, mse, mae        = run_svr_arima_svr(train, val, test)
        results["SVR(ARIMA,SVR)"]  = {"MSE": mse, "MAE": mae}
        preds_dict["SVR(ARIMA,SVR)"] = sas_preds

        # 8) Modelo proposto (Etapa 4)
        results["Proposto"] = PROPOSED_RESULTS[ds_name]
        preds_dict["Proposto"] = []  # sem preds salvas aqui

        all_results[ds_name] = results

        # Grafico comparativo
        plot_all_models(test, preds_dict, ds_name)

    # ------------------------------------------
    # TABELAS FINAIS COMPLETAS
    # Equivalente Tabelas 5, 6 e 7 do artigo
    # ------------------------------------------
    MODEL_ORDER = ["ARIMA", "MLP", "SVR", "DTR",
                   "ARIMA+MLP", "ARIMA+SVR", "SVR(ARIMA,SVR)", "Proposto"]

    # Tabela MSE
    mse_rows = []
    for ds in DATASETS.keys():
        if ds not in all_results:
            continue
        row = {"Dataset": ds}
        for m in MODEL_ORDER:
            row[m] = all_results[ds][m]["MSE"]
        mse_rows.append(row)
    df_mse = pd.DataFrame(mse_rows).set_index("Dataset")

    # Tabela MAE
    mae_rows = []
    for ds in DATASETS.keys():
        if ds not in all_results:
            continue
        row = {"Dataset": ds}
        for m in MODEL_ORDER:
            row[m] = all_results[ds][m]["MAE"]
        mae_rows.append(row)
    df_mae = pd.DataFrame(mae_rows).set_index("Dataset")

    # Tabela Ganho (ARIMA como referencia)
    gain_rows = []
    for ds in DATASETS.keys():
        if ds not in all_results:
            continue
        row      = {"Dataset": ds}
        arima_mse = all_results[ds]["ARIMA"]["MSE"]
        for m in MODEL_ORDER:
            gain    = (arima_mse - all_results[ds][m]["MSE"]) / arima_mse * 100
            row[m]  = round(gain, 2)
        gain_rows.append(row)
    df_gain = pd.DataFrame(gain_rows).set_index("Dataset")

    print("\n" + "=" * 70)
    print("  TABELA MSE COMPLETA (equiv. Tabela 5 do artigo)")
    print("  [DADOS SINTETICOS]")
    print("=" * 70)
    print(df_mse.T.to_string())

    print("\n" + "=" * 70)
    print("  TABELA MAE COMPLETA (equiv. Tabela 6 do artigo)")
    print("  [DADOS SINTETICOS]")
    print("=" * 70)
    print(df_mae.T.to_string())

    print("\n" + "=" * 70)
    print("  TABELA DE GANHO % vs ARIMA (equiv. Tabela 7 do artigo)")
    print("  Positivo = melhor que ARIMA | Negativo = pior que ARIMA")
    print("=" * 70)
    print(df_gain.T.to_string())

    # Salva CSVs
    df_mse.T.to_csv(os.path.join(RESULTS_PATH, "FINAL_tabela_MSE.csv"))
    df_mae.T.to_csv(os.path.join(RESULTS_PATH, "FINAL_tabela_MAE.csv"))
    df_gain.T.to_csv(os.path.join(RESULTS_PATH, "FINAL_tabela_ganho.csv"))

    print("\nCSVs salvos em results/FINAL_tabela_*.csv")
    print("\n" + "=" * 55)
    print("  Etapa 5 concluida!")
    print("  Proximo passo: 06_nemenyi_test.py")
    print("=" * 55)