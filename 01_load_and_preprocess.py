# 01_load_and_preprocess.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# CONFIGURACOES GLOBAIS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# FUNCAO: Normalizacao Min-Max  (Eq. 14 do artigo)
# ─────────────────────────────────────────────
def minmax_normalize(series: np.ndarray):
    """
    Z_norm = (Z - min(Z)) / (max(Z) - min(Z))
    Retorna: serie normalizada, valor minimo e maximo (para desnormalizar depois)
    """
    z_min = series.min()
    z_max = series.max()
    normalized = (series - z_min) / (z_max - z_min)
    return normalized, z_min, z_max


def minmax_denormalize(series_norm: np.ndarray, z_min: float, z_max: float):
    """Reverte a normalizacao para comparacao na escala original."""
    return series_norm * (z_max - z_min) + z_min


# ─────────────────────────────────────────────
# FUNCAO: Divisao treino / validacao / teste
# ─────────────────────────────────────────────
def split_dataset(series: np.ndarray, train_ratio=0.60,
                  val_ratio=0.20, test_ratio=0.20):
    """
    Divide a serie temporal respeitando a ordem cronologica.
    Proporcoes: 60% treino | 20% validacao | 20% teste  (Tabela 2 do artigo)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "As proporcoes devem somar 1."

    n = len(series)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    train = series[:i_train]
    val   = series[i_train:i_val]
    test  = series[i_val:]

    return train, val, test


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL DE PRE-PROCESSAMENTO
# ─────────────────────────────────────────────
def load_and_preprocess(csv_filename: str, dataset_name: str, target_col: str = "SoC"):
    """
    Carrega um CSV sintetico, normaliza e divide em subconjuntos.

    NOTA SOBRE DADOS SINTETICOS:
    Como os dados originais nao estao disponiveis, utilizamos series sinteticas
    que preservam as propriedades estatisticas gerais das series de SoC
    (ciclicidade, degradacao gradual, ruido gaussiano). Resultados numericos
    podem diferir dos relatados no artigo, mas o comportamento metodologico
    deve ser reproduzivel.
    """
    filepath = os.path.join(DATA_PATH, csv_filename)

    # Leitura do CSV
    df = pd.read_csv(filepath)

    # Busca a coluna alvo de forma case-insensitive
    col_map = {c.lower(): c for c in df.columns}
    if target_col.lower() not in col_map:
        raise ValueError(
            f"Coluna '{target_col}' nao encontrada em {csv_filename}.\n"
            f"Colunas disponiveis: {list(df.columns)}"
        )
    target_col_real = col_map[target_col.lower()]

    raw_series = df[target_col_real].dropna().values.astype(float)

    # Normalizacao (Eq. 14)
    norm_series, z_min, z_max = minmax_normalize(raw_series)

    # Divisao
    train, val, test = split_dataset(norm_series)

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}  |  Arquivo: {csv_filename}")
    print(f"  Total de amostras : {len(norm_series)}")
    print(f"  Treino            : {len(train)}  ({len(train)/len(norm_series)*100:.1f}%)")
    print(f"  Validacao         : {len(val)}    ({len(val)/len(norm_series)*100:.1f}%)")
    print(f"  Teste             : {len(test)}   ({len(test)/len(norm_series)*100:.1f}%)")
    print(f"  Min original      : {z_min:.4f}   Max original: {z_max:.4f}")
    print(f"{'='*50}")

    return {
        "name"       : dataset_name,
        "raw"        : raw_series,
        "normalized" : norm_series,
        "train"      : train,
        "val"        : val,
        "test"       : test,
        "z_min"      : z_min,
        "z_max"      : z_max,
    }


def visualize_split(dataset_dict: dict):
    """Plota a divisao da serie para visualizacao no relatorio."""
    d = dataset_dict
    n_train = len(d["train"])
    n_val   = len(d["val"])
    n_test  = len(d["test"])
    total   = len(d["normalized"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(n_train),
            d["train"], color="steelblue", label="Treino (60%)")
    ax.plot(range(n_train, n_train + n_val),
            d["val"],   color="orange",    label="Validacao (20%)")
    ax.plot(range(n_train + n_val, total),
            d["test"],  color="crimson",   label="Teste (20%)")

    ax.axvline(n_train,         color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(n_train + n_val, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"[DADOS SINTETICOS] {d['name']} - Divisao da serie temporal normalizada")
    ax.set_xlabel("Passo de tempo")
    ax.set_ylabel("SoC normalizado")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(RESULTS_PATH, f"{d['name']}_split.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Grafico salvo em: {save_path}")


# ─────────────────────────────────────────────
# EXECUCAO
# ─────────────────────────────────────────────
if __name__ == "__main__":

    all_datasets = {}

    for name, filename in DATASETS.items():
        try:
            data = load_and_preprocess(filename, name, target_col="SoC")
            visualize_split(data)
            all_datasets[name] = data
        except FileNotFoundError:
            print(f"AVISO: Arquivo nao encontrado: {filename} - pulando.")
        except ValueError as e:
            print(f"AVISO: Erro no dataset {name}: {e}")

    print("\nPre-processamento concluido para todos os datasets disponiveis.")