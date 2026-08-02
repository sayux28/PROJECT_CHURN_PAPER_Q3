"""
Preprocesamiento del dataset de churn de telecomunicaciones
(Maven Analytics / IBM, via kagglehub: shilongzhuang/telecom-customer-churn-by-maven-analytics)

Cubre todo lo que va DESPUÉS del EDA exploratorio y ANTES de modelar:
    1. Carga de datos
    2. Definición del target (excluye clientes "Joined")
    3. Split train/test estratificado
    4. Imputación estructural (fit-free, reglas fijas del diccionario de datos)
    5. Corrección de valores inválidos (Monthly Charge negativo)
    6. Selección de columnas (drop de ID, geografía cruda y leakage)
    7. Escalado + encoding con ColumnTransformer (fit SOLO en train)
    8. Export de las matrices listas para modelar

Toda la salida de consola se guarda además en docs/result_eda_churn_v1.txt
(se sobrescribe en cada corrida).

Uso:
    python eda_churn_v1.py
    python eda_churn_v1.py --churn-csv ruta/telecom_customer_churn.csv --out-dir ruta/salida
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class _Tee:
    """Escribe cada línea impresa en varios streams a la vez (consola + archivo de log)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80/20, replica el protocolo del paper SHAP-Rule (Sección 4.3)

# Columnas que quedan vacías por diseño (según el diccionario de datos) y no
# representan datos perdidos: se imputan con una categoría/valor fijo, no con
# un estadístico, así que no hay riesgo de leakage al aplicarlas después del split.
COLS_SIN_INTERNET = [
    "Internet Type", "Online Security", "Online Backup", "Device Protection Plan",
    "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music",
    "Unlimited Data",
]

# Identificador, geografía cruda (alta cardinalidad / redundante con lat-long)
# y columnas con leakage del target: no pueden entrar como features.
DROP_COLS = [
    "Customer ID", "City", "Zip Code", "Latitude", "Longitude",
    "Churn Category", "Churn Reason", "Customer Status",
]


def load_data(
    churn_csv: str | None = None, population_csv: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset de churn (38 columnas, Maven Analytics) y el de
    población por código postal: kagglehub si está disponible, si no CSV
    locales. 'pop' no se usa en el preprocesamiento (Zip Code/geografía se
    excluyen en DROP_COLS), queda disponible para análisis geográfico aparte.
    """
    if churn_csv and population_csv:
        return pd.read_csv(churn_csv), pd.read_csv(population_csv)
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "shilongzhuang/telecom-customer-churn-by-maven-analytics"
        )
        df = pd.read_csv(f"{path}/telecom_customer_churn.csv")
        pop = pd.read_csv(f"{path}/telecom_zipcode_population.csv")
        return df, pop
    except Exception as e:
        print(f"kagglehub no disponible ({e}); se buscan los CSV localmente.")
        df = pd.read_csv(churn_csv or "telecom_customer_churn.csv")
        pop = pd.read_csv(population_csv or "telecom_zipcode_population.csv")
        return df, pop


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define Churn_Flag y excluye a los clientes 'Joined'.

    'Joined' son clientes que recién se dieron de alta este trimestre: su tenure
    es casi 0 y todavía no tuvieron chance de irse. Si se dejan en el dataset de
    modelado quedan codificados como "no churn" (Churn_Flag=0) igual que 'Stayed',
    contaminando el target con clientes cuyo desenlace real aún se desconoce.
    Por eso se excluyen ANTES del split, y el target queda binario y limpio.
    """
    df = df.copy()
    antes = len(df)
    df = df[df["Customer Status"].isin(["Stayed", "Churned"])].copy()
    excluidos = antes - len(df)
    print(f"Clientes 'Joined' excluidos del modelado: {excluidos} (de {antes})")

    df["Churn_Flag"] = (df["Customer Status"] == "Churned").astype(int)
    return df


def analizar_balance_clases(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Cuantifica el desbalance de clases de Churn_Flag (después de excluir 'Joined'):
    conteo y porcentaje por clase + ratio mayoría/minoría en texto, y el mismo
    conteo en un bar chart guardado en docs/.
    """
    etiquetas = {0: "Stayed (0)", 1: "Churned (1)"}
    conteo = df["Churn_Flag"].value_counts().sort_index()
    porcentaje = conteo / conteo.sum() * 100
    ratio = conteo.max() / conteo.min()

    print("Balance de clases (Churn_Flag, tras excluir 'Joined'):")
    for clase, n in conteo.items():
        print(f"  {etiquetas[clase]}: {n} ({porcentaje[clase]:.1f}%)")
    print(f"  Ratio de desbalance (mayoría:minoría) = {ratio:.2f} : 1")

    colores = {0: "#2a78d6", 1: "#e34948"}  # azul=Stayed, rojo=Churned (paleta fija del proyecto)
    fig, ax = plt.subplots(figsize=(5, 4))
    barras = ax.bar(
        [etiquetas[c] for c in conteo.index],
        conteo.values,
        color=[colores[c] for c in conteo.index],
    )
    for barra, n, pct in zip(barras, conteo.values, porcentaje.values):
        ax.annotate(
            f"{n}\n({pct:.1f}%)",
            xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("N° de clientes")
    ax.set_ylim(0, conteo.max() * 1.15)
    ax.set_title("Balance de clases: Churn_Flag (sin 'Joined')", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "balance_clases_churn.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico de balance de clases guardado en: {fig_path}")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Churn_Flag"]
    )
    print(f"Train: {df_train.shape} | Test: {df_test.shape}")
    print(
        f"Tasa de churn -> train: {df_train['Churn_Flag'].mean():.3f} "
        f"| test: {df_test['Churn_Flag'].mean():.3f}"
    )
    return df_train, df_test


def imputar_estructural(d: pd.DataFrame) -> pd.DataFrame:
    """Reglas fijas del diccionario de datos (no estadísticos): idéntico en train y test."""
    d = d.copy()
    d[COLS_SIN_INTERNET] = d[COLS_SIN_INTERNET].fillna("No Internet Service")
    d["Avg Monthly GB Download"] = d["Avg Monthly GB Download"].fillna(0)
    d["Multiple Lines"] = d["Multiple Lines"].fillna("No Phone Service")
    d["Avg Monthly Long Distance Charges"] = d["Avg Monthly Long Distance Charges"].fillna(0)
    d["Offer"] = d["Offer"].fillna("None")
    return d


def corregir_monthly_charge(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    'Monthly Charge' negativo no tiene interpretación de negocio válida (los
    créditos/reembolsos ya se registran aparte en 'Total Refunds'), así que se
    trata como error de captura y se corrige con la mediana de los valores
    válidos... calculada SOLO en train, para no filtrar información de test.
    """
    df_train, df_test = df_train.copy(), df_test.copy()
    mediana = df_train.loc[df_train["Monthly Charge"] >= 0, "Monthly Charge"].median()

    n_train = int((df_train["Monthly Charge"] < 0).sum())
    n_test = int((df_test["Monthly Charge"] < 0).sum())
    df_train.loc[df_train["Monthly Charge"] < 0, "Monthly Charge"] = mediana
    df_test.loc[df_test["Monthly Charge"] < 0, "Monthly Charge"] = mediana

    print(f"Monthly Charge negativo corregido con mediana de train ({mediana:.2f}): "
          f"train={n_train} filas, test={n_test} filas")
    return df_train, df_test


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df["Churn_Flag"]
    X = df.drop(columns=DROP_COLS + ["Churn_Flag"])
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_feats = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X_train.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Numéricas: {len(num_feats)} | Categóricas: {len(cat_feats)}")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            # drop='if_binary' evita generar dos columnas redundantes (Yes/No)
            # para las variables que sólo tienen dos categorías.
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_feats),
        ],
        verbose_feature_names_out=False,
    )


def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame):
    X_train, y_train = get_X_y(df_train)
    X_test, y_test = get_X_y(df_test)

    preprocessor = build_preprocessor(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=X_test.index)

    print(f"X_train_prep: {X_train_prep.shape} | X_test_prep: {X_test_prep.shape}")
    assert X_train_prep.isnull().sum().sum() == 0, "Quedaron nulos en train tras el preprocesamiento"
    assert X_test_prep.isnull().sum().sum() == 0, "Quedaron nulos en test tras el preprocesamiento"

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor


def save_outputs(out_dir: Path, X_train_prep, X_test_prep, y_train, y_test) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train_prep.to_csv(out_dir / "X_train_prep.csv", index=False)
    X_test_prep.to_csv(out_dir / "X_test_prep.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    print(f"Matrices guardadas en: {out_dir}")


def run_pipeline(args) -> None:
    df, pop = load_data(args.churn_csv, args.population_csv)
    print(f"Dataset original: {df.shape} | Población por zip: {pop.shape}")

    df = build_target(df)
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    analizar_balance_clases(df, docs_dir)
    df_train, df_test = split_data(df)

    df_train = imputar_estructural(df_train)
    df_test = imputar_estructural(df_test)
    df_train, df_test = corregir_monthly_charge(df_train, df_test)

    for nombre, d in [("train", df_train), ("test", df_test)]:
        nulos_relevantes = d.drop(columns=["Churn Category", "Churn Reason"]).isnull().sum().sum()
        print(f"{nombre}: nulos fuera de leakage cols = {int(nulos_relevantes)}")

    X_train_prep, X_test_prep, y_train, y_test, _ = preprocess(df_train, df_test)

    save_outputs(Path(args.out_dir), X_train_prep, X_test_prep, y_train, y_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--churn-csv", default=None, help="Ruta local a telecom_customer_churn.csv")
    parser.add_argument(
        "--population-csv", default=None, help="Ruta local a telecom_zipcode_population.csv"
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "processed"),
        help="Carpeta donde se guardan las matrices preprocesadas",
    )
    parser.add_argument(
        "--log-file",
        default=str(Path(__file__).resolve().parent.parent / "docs" / "result_eda_churn_v1.txt"),
        help="Archivo .txt donde se guarda una copia de todo lo impreso en consola",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
            run_pipeline(args)

    print(f"Log de ejecución guardado en: {log_path}")


if __name__ == "__main__":
    main()
