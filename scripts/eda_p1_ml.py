"""
eda_p1_ml.py — pipeline único que compara 4 modelos de clasificación sobre
el dataset de churn: Árbol de Decisión, Random Forest, LightGBM (árboles) y
Regresión Logística (modelo lineal).

Antes vivía en dos scripts separados (eda_p1_ml.py + eda_p2_lineal.py) que
duplicaban toda la carga/limpieza/diagnóstico de datos. Ahora ese tramo
común corre UNA sola vez, y el pipeline se bifurca recién en el paso de
codificación de categóricas, porque cada familia de modelos necesita un
preprocesamiento distinto e incompatible entre sí:

  - Árboles (Decision Tree, Random Forest, LightGBM): categóricas con
    codificación ORDINAL (un entero por categoría) y numéricas SIN escalar.
    Los árboles dividen por umbral, así que son invariantes a la escala, y
    el orden arbitrario que impone un entero no les afecta (no asumen que
    "2" es "el doble" que "1").
  - Regresión Logística: categóricas con ONE-HOT (`drop='if_binary'`) y
    numéricas con StandardScaler. Un modelo lineal SÍ es sensible a
    orden/magnitud: la codificación ordinal le haría creer que, por
    ejemplo, "Fiber Optic" (código 2) vale el doble que "DSL" (código 1),
    lo cual no tiene sentido para una variable sin orden real.

Por eso cada preprocesamiento se guarda en su propia carpeta
(`data/processed/` para árboles, `data/processed_linear/` para la
regresión logística) y nunca se mezclan las matrices resultantes: cada
modelo se entrena y evalúa solo con la matriz que le corresponde. Los
pasos que SÍ son independientes de la codificación (diagnóstico de
Monthly Charge, definición de target, split, imputación estructural,
análisis de numéricas y eliminación por VIF) se calculan una sola vez y
se reutilizan en ambas ramas, así no hay resultados contradictorios entre
lo que ve un árbol y lo que ve la regresión logística.

PARTE 3 entrena los 4 modelos con una única configuración simple (modelo
"inicial"), para explicar su comportamiento sobre este dataset. PARTE 4
ajusta hiperparámetros SOLO de los 3 árboles (la Regresión Logística queda
igual, sin tocar): Árbol de Decisión con poda por costo-complejidad
(ccp_alpha) + GridSearchCV puntual (pocos hiperparámetros, entrena en
milisegundos); Random Forest y LightGBM con Optuna/TPE (espacios de
hiperparámetros más grandes, donde la búsqueda bayesiana explora mejor que
un grid exhaustivo). Detalle de la estrategia en
docs/plan_shap_rule_churn_telecom.md (Fase 4b).

Toda la salida (impresa en consola) se guarda además en:
    - docs/hallazgos_numericas_p1.txt (diagnóstico de numéricas + VIF)
    - docs/resultado_modelos_ml.txt (métricas de los 4 modelos iniciales +
      ajuste de hiperparámetros de los árboles + comparación inicial vs.
      ajustado)

Cómo ejecutarlo por partes en VS Code: abrir el archivo, correr cada
bloque "# %%" con "Run Cell" en la Python Interactive Window. También
corre completo por terminal: python eda_p1_ml.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Asegura que 'eda_churn_v1' se pueda importar sin importar desde dónde se
# lance la celda (VS Code normalmente ya agrega la carpeta del archivo).
sys.path.append(str(Path(__file__).resolve().parent))

# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

from eda_churn_v1 import (
    RANDOM_STATE,
    build_target,
    corregir_monthly_charge,
    get_X_y,
    imputar_estructural,
    load_data,
    preprocess,
    save_outputs,
    split_data,
)

# ============================================================================
# PARTE 1: pipeline común (idéntico para todos los modelos, corre una sola
# vez). Nada de lo que hay aquí depende de qué modelo se vaya a entrenar
# después.
# ============================================================================

# %% Cargar dataset completo (kagglehub si está disponible, si no CSV local)
df, pop = load_data()
print(f"Dataset original: {df.shape} | Población por zip: {pop.shape}")

# %% Diagnóstico: Monthly Charge negativo (descriptivo, sobre TODO el dataset)


def analizar_monthly_charge_negativo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cuenta y muestra los registros con 'Monthly Charge' < 0.

    Es puramente descriptivo (no se calcula ni aplica ningún estadístico
    aquí), por eso se corre sobre el dataset completo, antes del split. La
    corrección real (imputar con la mediana de train) sigue viviendo en
    corregir_monthly_charge(), que se aplica después del split para no
    filtrar información de test.
    """
    cols = ["Monthly Charge", "Total Charges", "Tenure in Months", "Customer Status"]
    negativos = df.loc[df["Monthly Charge"] < 0, cols]
    print(f"Monthly Charge negativos: {len(negativos)}")
    print(negativos)
    return negativos


negativos_mc = analizar_monthly_charge_negativo(df)

# %% Diagnóstico: Tenure (antigüedad) por Customer Status — valida los 3 grupos
# (Stayed/Churned/Joined) ANTES de que build_target() excluya a 'Joined'.


def graficar_tenure_por_status(df: pd.DataFrame, ruta: Path) -> pd.DataFrame:
    """Boxplot de Tenure in Months por Customer Status, sobre el dataset completo. Detalle en docs/doc_funcional.md."""
    orden = ["Joined", "Churned", "Stayed"]
    datos = [df.loc[df["Customer Status"] == s, "Tenure in Months"] for s in orden]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(datos, tick_labels=orden, vert=False)
    ax.set_xlabel("Tenure in Months")
    ax.set_title("Antigüedad (Tenure) por Customer Status")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en: {ruta}")

    resumen = df.groupby("Customer Status")["Tenure in Months"].describe()[["min", "25%", "50%", "75%", "max"]]
    print(resumen)

    churn_temprano = df[(df["Customer Status"] == "Churned") & (df["Tenure in Months"] <= 2)]
    print(f"\nClientes Churned con Tenure <= 2 meses: {len(churn_temprano)}")
    return resumen


tenure_status_resumen = graficar_tenure_por_status(
    df, Path(__file__).resolve().parent.parent / "docs" / "tenure_por_status.png"
)

# %% Definir target (excluye 'Joined', crea Churn_Flag)
df = build_target(df)

# %% Split train/test estratificado
df_train, df_test = split_data(df)

# %% Imputación estructural (reglas fijas, sin leakage)
df_train = imputar_estructural(df_train)
df_test = imputar_estructural(df_test)

# %% Corrección de Monthly Charge negativo (mediana de TRAIN, aplicada a ambos)
df_train, df_test = corregir_monthly_charge(df_train, df_test)

# %% Listar variables numéricas / categóricas. La codificación de las
# categóricas se decide más abajo, por familia de modelo (ver PARTE 2):
# ordinal para árboles, one-hot + StandardScaler para regresión logística.


def resumen_variables(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Lista numéricas y categóricas de X. Detalle y por qué en docs/doc_funcional.md."""
    num_feats = X.select_dtypes(include=np.number).columns.tolist()
    cat_feats = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"Numéricas ({len(num_feats)}):")
    for c in num_feats:
        print(f"  - {c}")

    print(f"\nCategóricas ({len(cat_feats)}) -> codificación según el modelo (ver PARTE 2):")
    for c in cat_feats:
        print(f"  - {c} ({X[c].nunique()} categorías: {sorted(X[c].unique().tolist())})")

    return num_feats, cat_feats


X_train_raw, y_train_raw = get_X_y(df_train)
num_feats, cat_feats = resumen_variables(X_train_raw)

# %% Helper: imprime Y guarda en un .txt, para que los hallazgos queden
# visibles sin importar cómo se corra el archivo (Run Cell, Run All o
# terminal). La primera vez que se llama en la corrida sobrescribe el
# archivo; las siguientes veces agrega, así queda todo junto y en orden.

RUTA_HALLAZGOS = Path(__file__).resolve().parent.parent / "docs" / "hallazgos_numericas_p1.txt"
_log_iniciado = False


def _log(texto: str) -> None:
    global _log_iniciado
    print(texto)
    RUTA_HALLAZGOS.parent.mkdir(parents=True, exist_ok=True)
    modo = "w" if not _log_iniciado else "a"
    with open(RUTA_HALLAZGOS, modo, encoding="utf-8") as f:
        f.write(texto + "\n")
    _log_iniciado = True


# %% Varianza casi nula (numéricas casi constantes, aportan poca señal)


def detectar_varianza_casi_nula(
    X_train: pd.DataFrame, num_feats: list[str], umbral: float = 0.90
) -> pd.DataFrame:
    """% de filas con el valor más frecuente por variable. Diagnóstico, no elimina nada. Detalle en docs/doc_funcional.md."""
    filas = [
        {
            "variable": c,
            "valor_mas_frecuente": X_train[c].value_counts().index[0],
            "proporcion": round(X_train[c].value_counts(normalize=True).iloc[0], 3),
        }
        for c in num_feats
    ]
    resumen = pd.DataFrame(filas).sort_values("proporcion", ascending=False)

    _log("=== VARIANZA CASI NULA ===")
    _log(resumen.to_string(index=False))

    candidatas = resumen.loc[resumen["proporcion"] >= umbral, "variable"].tolist()
    if candidatas:
        _log(f"\nCandidatas a varianza casi nula (>= {umbral:.0%}): {candidatas}")
    else:
        _log(f"\nNinguna numérica supera el umbral de {umbral:.0%}.")
    return resumen


varianza_resumen = detectar_varianza_casi_nula(X_train_raw, num_feats)

# %% Estadísticos descriptivos y sesgo (media vs mediana, skew)


def resumen_descriptivo_numericas(X_train: pd.DataFrame, num_feats: list[str]) -> pd.DataFrame:
    """Media, mediana y sesgo (skew) por numérica. Insumo para SHAP-Rule Fase 5. Detalle en docs/doc_funcional.md."""
    tabla = pd.DataFrame(
        {
            "media": X_train[num_feats].mean(),
            "mediana": X_train[num_feats].median(),
            "skew": X_train[num_feats].skew(),
        }
    ).round(2)
    tabla["forma"] = tabla["skew"].apply(
        lambda s: "Simétrica" if abs(s) < 0.5 else ("Cola derecha" if s > 0 else "Cola izquierda")
    )
    _log("\n=== DESCRIPTIVOS Y SESGO ===")
    _log(tabla.to_string())
    return tabla


descriptivos = resumen_descriptivo_numericas(X_train_raw, num_feats)

# %% Outliers por IQR (regla de Tukey 1.5*IQR, calculado solo en train)


def detectar_outliers_iqr(X_train: pd.DataFrame, num_feats: list[str]) -> pd.DataFrame:
    """Outliers por Tukey 1.5*IQR (train). Diagnóstico, no elimina filas. Ojo con variables de muchos ceros: ver docs/doc_funcional.md."""
    filas = []
    for c in num_feats:
        q1, q3 = X_train[c].quantile(0.25), X_train[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((X_train[c] < lo) | (X_train[c] > hi)).sum())
        filas.append(
            {
                "variable": c,
                "limite_inferior": round(lo, 2),
                "limite_superior": round(hi, 2),
                "n_outliers": n_outliers,
                "pct": round(100 * n_outliers / len(X_train), 2),
            }
        )
    resumen = pd.DataFrame(filas).sort_values("n_outliers", ascending=False)
    _log("\n=== OUTLIERS POR IQR (Tukey 1.5*IQR) ===")
    _log(resumen.to_string(index=False))
    return resumen


outliers_resumen = detectar_outliers_iqr(X_train_raw, num_feats)

# %% Matriz de correlación (pares con |r| alto = posible redundancia)


def matriz_correlacion(X_train: pd.DataFrame, num_feats: list[str], umbral: float = 0.7) -> pd.DataFrame:
    """Correlación de Pearson entre numéricas (train) y pares con |r| >= umbral. Solo ve pares, no redundancia de 3+ (para eso, VIF)."""
    corr = X_train[num_feats].corr()
    _log("\n=== CORRELACIÓN DE PEARSON ===")
    _log(corr.round(2).to_string())

    _log(f"\nPares con |r| >= {umbral}:")
    for i, c1 in enumerate(num_feats):
        for c2 in num_feats[i + 1 :]:
            r = corr.loc[c1, c2]
            if abs(r) >= umbral:
                _log(f"  {c1} <-> {c2}: r = {r:.3f}")
    return corr


def graficar_matriz_correlacion(corr: pd.DataFrame, ruta: Path) -> None:
    """Heatmap de la matriz de correlación, guardado como imagen de referencia (no requiere seaborn)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="r de Pearson")
    ax.set_title("Matriz de correlación de Pearson (numéricas, train)")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en: {ruta}")


correlacion = matriz_correlacion(X_train_raw, num_feats)
graficar_matriz_correlacion(
    correlacion, Path(__file__).resolve().parent.parent / "docs" / "matriz_correlacion_pearson.png"
)

# %% VIF (Variance Inflation Factor) y eliminación iterativa de multicolineales.
# Se calcula UNA sola vez porque solo depende de las numéricas crudas, antes
# de codificar categóricas -- el resultado (numericas_finales) es el mismo
# sin importar si después se entrena un árbol o una regresión logística.


def calcular_vif(X_train: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """VIF = 1/(1-R²) de cada variable regresionada contra el resto. VIF alto = redundante. Detalle en docs/doc_funcional.md."""
    filas = []
    for c in feats:
        otras = [f for f in feats if f != c]
        modelo = LinearRegression().fit(X_train[otras], X_train[c])
        r2 = modelo.score(X_train[otras], X_train[c])
        vif = np.inf if r2 >= 0.9999 else 1 / (1 - r2)
        filas.append({"variable": c, "r2": round(r2, 4), "VIF": vif})
    return pd.DataFrame(filas).sort_values("VIF", ascending=False, kind="stable")


def eliminar_multicolineales_vif(
    X_train: pd.DataFrame, num_feats: list[str], umbral_vif: float = 10.0
) -> list[str]:
    """Elimina iterativamente la numérica con mayor VIF hasta que todas queden bajo umbral_vif. No usa PCA (rompe trazabilidad para SHAP-Rule). Detalle en docs/doc_funcional.md."""
    restantes = list(num_feats)
    paso = 1
    while len(restantes) > 1:
        tabla = calcular_vif(X_train, restantes)
        _log(f"\n--- VIF paso {paso} ---")
        _log(tabla.to_string(index=False))
        peor = tabla.iloc[0]
        if peor["VIF"] <= umbral_vif:
            break
        _log(f"Se elimina '{peor['variable']}' (VIF = {peor['VIF']})")
        restantes.remove(peor["variable"])
        paso += 1
    return restantes


numericas_finales = eliminar_multicolineales_vif(X_train_raw, num_feats, umbral_vif=10.0)
eliminadas_vif = [c for c in num_feats if c not in numericas_finales]

_log("\n=== RESUMEN DE HALLAZGOS (numéricas) ===")
_log(f"- Varianza casi nula: {varianza_resumen.loc[varianza_resumen['proporcion'] >= 0.90, 'variable'].tolist()} (se conserva, es diagnóstico)")
_log(f"- Outliers IQR más altos: {outliers_resumen.head(3)['variable'].tolist()} (variables con muchos ceros, no son errores)")
_log(f"- Multicolinealidad (VIF > 10): eliminadas {eliminadas_vif}")
_log(f"- Numéricas finales: {len(numericas_finales)} de {len(num_feats)} -> {numericas_finales}")
_log(f"- Gráfico de correlación: docs/matriz_correlacion_pearson.png")
_log(f"- Detalle y explicación de cada punto: docs/doc_funcional.md")

# %% Distribución de clases y estrategia de desbalance (aplica a los 4 modelos:
# class_weight='balanced' para árboles de sklearn y regresión logística,
# scale_pos_weight para LightGBM -- nunca ambas técnicas en el mismo modelo).

conteo_clases = y_train_raw.value_counts()
print(f"\nDistribución del target (train): {conteo_clases.to_dict()}")
print(
    "Desbalance moderado (no severo) -> se usa class_weight='balanced' (árboles de "
    "sklearn y regresión logística) o scale_pos_weight (LightGBM), NO SMOTE: SMOTE "
    "interpolaría de forma inválida entre categorías (one-hot o enteros ordinales), "
    "y el desbalance no es lo bastante severo para justificarlo."
)

# ============================================================================
# PARTE 2: a partir de aquí el pipeline se bifurca por familia de modelo.
# Cada rama tiene su propia codificación de categóricas y sus propias
# matrices de salida -- nunca se mezclan entre sí.
# ============================================================================

# %% Rama A (árboles): codificación ordinal, numéricas sin escalar.
# Ver docstring del módulo y docs/plan_shap_rule_churn_telecom.md.


def convertir_categoricas_ordinal(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cat_feats: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Codificación ordinal (un entero por categoría), fit solo en train. Estrategia SHAP-Rule: ver docs/plan_shap_rule_churn_telecom.md."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    df_train_enc = df_train.copy()
    df_test_enc = df_test.copy()
    df_train_enc[cat_feats] = encoder.fit_transform(df_train[cat_feats])
    df_test_enc[cat_feats] = encoder.transform(df_test[cat_feats])

    return df_train_enc, df_test_enc, encoder


def resumen_conversion_categoricas(
    cat_feats: list[str],
    encoder: OrdinalEncoder,
    df_original: pd.DataFrame,
    df_codificado: pd.DataFrame,
) -> None:
    """Muestra el mapeo categoría->código aprendido en train y una vista antes/después."""
    print("Mapeo categoría -> código (aprendido solo con train):")
    for col, categorias in zip(cat_feats, encoder.categories_):
        print(f"\n{col}:")
        for codigo, categoria in enumerate(categorias):
            print(f"  {categoria!r} -> {codigo}")

    print("\nPrimeras 5 filas - ANTES (categórico original):")
    print(df_original[cat_feats].head(5))

    print("\nPrimeras 5 filas - DESPUÉS (codificado ordinal):")
    print(df_codificado[cat_feats].head(5))


df_train_ord, df_test_ord, ordinal_encoder = convertir_categoricas_ordinal(df_train, df_test, cat_feats)
resumen_conversion_categoricas(cat_feats, ordinal_encoder, df_train, df_train_ord)

X_train_ord_full, y_train = get_X_y(df_train_ord)
X_test_ord_full, y_test = get_X_y(df_test_ord)

columnas_finales = numericas_finales + cat_feats
X_train_arbol = X_train_ord_full[columnas_finales]
X_test_arbol = X_test_ord_full[columnas_finales]
print(f"[Árboles] X_train: {X_train_arbol.shape} | X_test: {X_test_arbol.shape}")

save_outputs(
    Path(__file__).resolve().parent.parent / "data" / "processed",
    X_train_arbol,
    X_test_arbol,
    y_train,
    y_test,
)

# %% Rama B (regresión logística): one-hot + StandardScaler.
# Se excluyen antes las numéricas eliminadas por VIF, así el
# ColumnTransformer (StandardScaler + OneHotEncoder de eda_churn_v1.preprocess)
# ya no las ve.

df_train_reducido = df_train.drop(columns=eliminadas_vif)
df_test_reducido = df_test.drop(columns=eliminadas_vif)

X_train_lineal, X_test_lineal, y_train_lineal, y_test_lineal, preprocessor_lineal = preprocess(
    df_train_reducido, df_test_reducido
)
print(f"[Regresión Logística] X_train: {X_train_lineal.shape} | X_test: {X_test_lineal.shape}")
# y_train_lineal/y_test_lineal son las mismas filas y valores que y_train/y_test
# (mismo split, misma columna Churn_Flag) -- se guardan aparte solo porque
# preprocess() las devuelve, no porque difieran.

save_outputs(
    Path(__file__).resolve().parent.parent / "data" / "processed_linear",
    X_train_lineal,
    X_test_lineal,
    y_train_lineal,
    y_test_lineal,
)

# ============================================================================
# PARTE 3: entrenamiento y comparación de los 4 modelos, cada uno con la
# matriz que le corresponde. Sin GridSearchCV -- una sola configuración
# simple por modelo (eso ya vive en eda_p1.py, Fase 4).
# ============================================================================

# %% Log dedicado a los resultados de modelado (consola + docs/resultado_modelos_ml.txt)

RUTA_RESULTADOS_ML = Path(__file__).resolve().parent.parent / "docs" / "resultado_modelos_ml.txt"
_log_ml_iniciado = False


def _log_ml(texto: str) -> None:
    global _log_ml_iniciado
    print(texto)
    RUTA_RESULTADOS_ML.parent.mkdir(parents=True, exist_ok=True)
    modo = "w" if not _log_ml_iniciado else "a"
    with open(RUTA_RESULTADOS_ML, modo, encoding="utf-8") as f:
        f.write(texto + "\n")
    _log_ml_iniciado = True


_log_ml("=== CÓMO FUNCIONA CADA MODELO (explicación breve) ===")
_log_ml(
    "\nÁrbol de Decisión: un único árbol que va haciendo preguntas sucesivas\n"
    "sobre las variables (ej. '¿Contract es Month-to-Month?'), eligiendo en\n"
    "cada nodo la pregunta que mejor separa Churn de No churn. Es el más\n"
    "simple e interpretable de los tres (se puede dibujar y leer directo),\n"
    "pero sin limitarlo tiende a memorizar el train (sobreajuste): aprende\n"
    "hasta el ruido de los datos de entrenamiento y generaliza peor en test."
)
_log_ml(
    "\nRandom Forest: entrena muchos árboles de decisión (aquí 100) sobre\n"
    "submuestras aleatorias de filas y de columnas (bagging), y promedia sus\n"
    "votos. Cada árbol individual puede sobreajustar, pero como ven porciones\n"
    "distintas de los datos, sus errores no están correlacionados entre sí y\n"
    "se cancelan al promediar -- por eso generaliza mejor que un árbol solo."
)
_log_ml(
    "\nLightGBM: también es un ensamble de árboles, pero construidos de forma\n"
    "SECUENCIAL en vez de en paralelo (gradient boosting): cada árbol nuevo\n"
    "se entrena específicamente para corregir los errores que dejó el\n"
    "conjunto de árboles anteriores. Suele lograr igual o mejor desempeño que\n"
    "Random Forest con menos árboles y entrena más rápido, a cambio de ser\n"
    "más sensible a sobreajustar si no se regula (learning_rate, profundidad)."
)
_log_ml(
    "\nRegresión Logística: estima la probabilidad de churn como una\n"
    "combinación lineal de las variables (ya escaladas con StandardScaler y\n"
    "codificadas con OneHotEncoder) pasada por una función sigmoide. A\n"
    "diferencia de los árboles, SÍ le importa la escala/orden de los números,\n"
    "por eso usa one-hot en vez de codificación ordinal. Es el más simple e\n"
    "interpretable en términos de coeficientes, pero solo captura relaciones\n"
    "lineales entre las variables (ya escaladas/codificadas) y el target."
)

_log_ml("\n=== QUÉ SIGNIFICA CADA MÉTRICA (para churn específicamente) ===")
_log_ml(
    "- Accuracy: % total de aciertos. Engañosa aquí por el desbalance de\n"
    "  clases (71.6% no-churn): un modelo que dijera \"nadie se va\" ya\n"
    "  tendría ~72% de accuracy sin detectar ni un solo churner.\n"
    "- Precision (Churn): de los clientes que el modelo marcó como \"se va a\n"
    "  ir\", ¿qué % realmente se fue? Importa para no gastar presupuesto de\n"
    "  retención en clientes que no pensaban irse.\n"
    "- Recall (Churn): de los clientes que SÍ se fueron, ¿qué % detectó el\n"
    "  modelo? Es la métrica más crítica del negocio -- cada churner no\n"
    "  detectado es un cliente perdido sin ningún intento de retención.\n"
    "- F1 (Churn): balance entre precision y recall (media armónica).\n"
    "- ROC-AUC: qué tan bien el modelo separa churn de no-churn en general,\n"
    "  sin depender de un umbral de decisión fijo (0.5 = azar, 1.0 = perfecto).\n"
    "- Accuracy train vs. test: la brecha entre ambas mide sobreajuste --\n"
    "  mientras más grande la diferencia, más memorizó el train en vez de\n"
    "  aprender un patrón que generaliza."
)

# %% Entrenar y evaluar un modelo: función reutilizable para los 4


def entrenar_y_evaluar(nombre: str, modelo, X_train, y_train, X_test, y_test) -> dict:
    """Entrena, predice y calcula las métricas principales de un modelo de clasificación binaria."""
    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - t0

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)
    proba_test = modelo.predict_proba(X_test)[:, 1]

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, pred_test, average="weighted")
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        y_test, pred_test, average="binary", pos_label=1
    )
    auc = roc_auc_score(y_test, proba_test)
    cm = confusion_matrix(y_test, pred_test)

    _log_ml(f"\n===== {nombre} =====")
    _log_ml(f"Tiempo de entrenamiento: {tiempo_entrenamiento:.3f} s")
    _log_ml(classification_report(y_test, pred_test, target_names=["No churn", "Churn"]))
    _log_ml(f"ROC-AUC: {auc:.3f}")
    _log_ml("Matriz de confusión (filas=real, columnas=predicho):")
    _log_ml("                 Pred. No churn  Pred. Churn")
    _log_ml(f"Real No churn   {cm[0, 0]:>14}  {cm[0, 1]:>11}")
    _log_ml(f"Real Churn      {cm[1, 0]:>14}  {cm[1, 1]:>11}")
    _log_ml(f"Accuracy train: {acc_train:.3f} | Accuracy test: {acc_test:.3f} | brecha: {acc_train - acc_test:.3f}")

    return {
        "Modelo": nombre,
        "Accuracy (train)": round(acc_train, 3),
        "Accuracy (test)": round(acc_test, 3),
        "Precision (Churn)": round(precision_c, 3),
        "Recall (Churn)": round(recall_c, 3),
        "F1 (Churn)": round(f1_c, 3),
        "Precision (w)": round(precision_w, 3),
        "Recall (w)": round(recall_w, 3),
        "F1 (w)": round(f1_w, 3),
        "ROC-AUC": round(auc, 3),
        "Tiempo (s)": round(tiempo_entrenamiento, 3),
    }


# %% Instanciar y correr los 4 modelos, cada uno con su matriz correspondiente
# (misma estrategia de desbalance que eda_p1.py: class_weight='balanced' para
# árboles de sklearn y regresión logística, scale_pos_weight para LightGBM)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (negativos/positivos): {scale_pos_weight:.3f}")

modelos_arbol = [
    (
        "Árbol de Decisión",
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
    ),
    (
        "Random Forest",
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_estimators=100),
    ),
    (
        "LightGBM",
        lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, scale_pos_weight=scale_pos_weight),
    ),
]

resultados = [
    entrenar_y_evaluar(nombre, modelo, X_train_arbol, y_train, X_test_arbol, y_test)
    for nombre, modelo in modelos_arbol
]

modelo_logistico = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
resultados.append(
    entrenar_y_evaluar(
        "Regresión Logística", modelo_logistico, X_train_lineal, y_train_lineal, X_test_lineal, y_test_lineal
    )
)

# %% Tabla comparativa (los 4 modelos) + interpretación automática de los resultados

tabla_comparativa = pd.DataFrame(resultados).set_index("Modelo")
_log_ml("\n=== TABLA COMPARATIVA (4 modelos) ===")
_log_ml(tabla_comparativa.to_string())

mejor_recall = tabla_comparativa["Recall (Churn)"].idxmax()
mejor_auc = tabla_comparativa["ROC-AUC"].idxmax()
mas_rapido = tabla_comparativa["Tiempo (s)"].idxmin()
brecha = tabla_comparativa["Accuracy (train)"] - tabla_comparativa["Accuracy (test)"]

_log_ml("\n=== INTERPRETACIÓN ===")
_log_ml(
    f"- Mejor recall detectando churn: {mejor_recall} "
    f"({tabla_comparativa.loc[mejor_recall, 'Recall (Churn)']}) -- el más relevante para "
    "el negocio, ya que minimiza clientes que se van sin ningún intento de retención."
)
_log_ml(
    f"- Mejor separación general de clases (ROC-AUC): {mejor_auc} "
    f"({tabla_comparativa.loc[mejor_auc, 'ROC-AUC']})."
)
_log_ml(f"- Entrenamiento más rápido: {mas_rapido} ({tabla_comparativa.loc[mas_rapido, 'Tiempo (s)']} s).")
_log_ml(
    "- Brecha train-test (sobreajuste) por modelo:\n"
    + "\n".join(f"    {modelo}: {valor:.3f}" for modelo, valor in brecha.items())
    + "\n  Una brecha mayor en el Árbol de Decisión confirma lo esperado: un solo árbol sin "
    "restricciones memoriza el train, mientras que los ensambles (RF, LightGBM) y el modelo "
    "lineal (Regresión Logística, con menos capacidad para sobreajustar) generalizan mejor."
)

# ============================================================================
# PARTE 4: ajuste de hiperparámetros de los árboles. La Regresión Logística
# NO se toca (queda con la configuración de PARTE 3). Estrategia por modelo
# (detalle y justificación en docs/plan_shap_rule_churn_telecom.md, Fase 4b):
#   - Árbol de Decisión: poda por costo-complejidad (ccp_alpha) + GridSearchCV
#     puntual (pocos hiperparámetros relevantes, entrena en milisegundos).
#   - Random Forest y LightGBM: Optuna (TPE/bayesiano) con 5-fold CV (espacios
#     de hiperparámetros más grandes, donde un grid exhaustivo escala mal).
# Los 3 optimizan F1 de la clase Churn (no el weighted): el desbalance de
# clases (~71.6% no-churn) diluye el weighted average con el desempeño sobre
# la clase mayoritaria, y optimizar solo recall llevaría a predecir churn
# indiscriminadamente.
# ============================================================================

CV_BUSQUEDA = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING_BUSQUEDA = "f1"  # F1 de la clase positiva (Churn=1) en clasificación binaria
N_TRIALS_OPTUNA = 30

_log_ml("\n\n=== AJUSTE DE HIPERPARÁMETROS (árboles) ===")
_log_ml(
    "Árbol de Decisión -> poda por costo-complejidad (ccp_alpha) + GridSearchCV puntual.\n"
    "Random Forest y LightGBM -> Optuna (TPE), 5-fold CV, "
    f"{N_TRIALS_OPTUNA} trials cada uno.\n"
    "Los 3 optimizan F1 de la clase Churn. La Regresión Logística no se toca."
)

# %% Árbol de Decisión: poda por costo-complejidad (ccp_alpha) + GridSearchCV


def tunear_arbol_ccp(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Busca el ccp_alpha óptimo (F1 Churn, 5-fold) sobre la ruta de poda del árbol sin restricciones."""
    arbol_sin_restricciones = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    ruta = arbol_sin_restricciones.cost_complexity_pruning_path(X_train, y_train)

    # Se descarta el último alpha (poda hasta un único nodo raíz, sin poder
    # predictivo) y se muestrea parejo si hay demasiados valores casi idénticos.
    ccp_alphas = np.unique(ruta.ccp_alphas)[:-1]
    if len(ccp_alphas) > 20:
        indices = np.linspace(0, len(ccp_alphas) - 1, 20, dtype=int)
        ccp_alphas = ccp_alphas[indices]

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        param_grid={"ccp_alpha": ccp_alphas},
        scoring=SCORING_BUSQUEDA,
        cv=CV_BUSQUEDA,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    _log_ml(
        f"\n[Árbol de Decisión] {len(ccp_alphas)} valores de ccp_alpha probados | "
        f"mejor ccp_alpha: {grid.best_params_['ccp_alpha']:.6f} | F1 (Churn) CV: {grid.best_score_:.3f}"
    )
    return grid.best_estimator_


arbol_tuned = tunear_arbol_ccp(X_train_arbol, y_train)

# %% Random Forest: Optuna (TPE), 5-fold CV


def objetivo_rf(trial: optuna.Trial) -> float:
    parametros = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    modelo = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", **parametros)
    scores = cross_val_score(modelo, X_train_arbol, y_train, scoring=SCORING_BUSQUEDA, cv=CV_BUSQUEDA, n_jobs=-1)
    return float(scores.mean())


study_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_rf.optimize(objetivo_rf, n_trials=N_TRIALS_OPTUNA)
_log_ml(
    f"\n[Random Forest] mejores hiperparámetros (Optuna, {N_TRIALS_OPTUNA} trials): {study_rf.best_params} | "
    f"F1 (Churn) CV: {study_rf.best_value:.3f}"
)
rf_tuned = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", **study_rf.best_params)

# %% LightGBM: Optuna (TPE), 5-fold CV, con EARLY STOPPING dentro de cada fold.
# A diferencia de RF (donde más árboles casi nunca perjudica), en boosting
# cada árbol nuevo corrige al anterior, así que agregar árboles de más SÍ
# sobreajusta -- por eso "n_estimators" ya no se busca como hiperparámetro
# fijo: se pone un tope alto (600) y se deja que early stopping corte en el
# punto óptimo por fold, mirando AUC sobre la porción de validación del fold
# (no vista por ese árbol). Esto evita que Optuna "pague" por más árboles de
# los que realmente ayudan, que es lo que explicaba el empeoramiento de
# LightGBM ajustado frente al inicial en las corridas anteriores (30 y 50
# trials sin early stopping).

STOPPING_ROUNDS = 30
TOPE_N_ESTIMATORS_LGBM = 600


def objetivo_lgbm(trial: optuna.Trial) -> float:
    parametros = {
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    f1_por_fold = []
    for idx_fit, idx_val in CV_BUSQUEDA.split(X_train_arbol, y_train):
        X_fit, X_val = X_train_arbol.iloc[idx_fit], X_train_arbol.iloc[idx_val]
        y_fit, y_val = y_train.iloc[idx_fit], y_train.iloc[idx_val]

        modelo = lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            scale_pos_weight=scale_pos_weight,
            n_estimators=TOPE_N_ESTIMATORS_LGBM,
            **parametros,
        )
        modelo.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=STOPPING_ROUNDS, verbose=False)],
        )
        pred_val = modelo.predict(X_val)
        f1_por_fold.append(f1_score(y_val, pred_val, pos_label=1))

    return float(np.mean(f1_por_fold))


study_lgbm = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_lgbm.optimize(objetivo_lgbm, n_trials=N_TRIALS_OPTUNA)
_log_ml(
    f"\n[LightGBM] mejores hiperparámetros (Optuna, {N_TRIALS_OPTUNA} trials, early stopping "
    f"por fold en AUC de validación, {STOPPING_ROUNDS} rondas de paciencia): {study_lgbm.best_params} | "
    f"F1 (Churn) CV: {study_lgbm.best_value:.3f}"
)

# El "n_estimators" ganador no viene de Optuna (no se buscó como hiperparámetro):
# se determina UNA vez más con early stopping, sobre una partición de
# validación dedicada tomada del train completo, para fijar el número de
# árboles del modelo final que se entrena sobre TODO X_train_arbol (donde ya
# no hay early stopping posible dentro de entrenar_y_evaluar).
X_fit_final, X_val_final, y_fit_final, y_val_final = train_test_split(
    X_train_arbol, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)
lgbm_early_stop = lgb.LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1,
    scale_pos_weight=scale_pos_weight,
    n_estimators=TOPE_N_ESTIMATORS_LGBM,
    **study_lgbm.best_params,
)
lgbm_early_stop.fit(
    X_fit_final,
    y_fit_final,
    eval_set=[(X_val_final, y_val_final)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=STOPPING_ROUNDS, verbose=False)],
)
n_estimators_lgbm_final = lgbm_early_stop.best_iteration_
_log_ml(
    f"[LightGBM] n_estimators final (early stopping sobre partición de validación dedicada, "
    f"tope {TOPE_N_ESTIMATORS_LGBM}): {n_estimators_lgbm_final}"
)

lgbm_params_finales = dict(study_lgbm.best_params, n_estimators=n_estimators_lgbm_final)
lgbm_tuned = lgb.LGBMClassifier(
    random_state=RANDOM_STATE, verbose=-1, scale_pos_weight=scale_pos_weight, **lgbm_params_finales
)

# %% Entrenar y evaluar los 3 árboles ajustados, con la misma función que los
# modelos iniciales (misma matriz X_train_arbol/X_test_arbol, mismas métricas)

modelos_arbol_tuned = [
    ("Árbol de Decisión (ajustado)", arbol_tuned),
    ("Random Forest (ajustado)", rf_tuned),
    ("LightGBM (ajustado)", lgbm_tuned),
]

resultados_tuned = [
    entrenar_y_evaluar(nombre, modelo, X_train_arbol, y_train, X_test_arbol, y_test)
    for nombre, modelo in modelos_arbol_tuned
]

# %% Hiperparámetros finales de cada modelo ajustado (tabla, para que queden
# visibles de un vistazo en docs/resultado_modelos_ml.txt, además del detalle
# ya impreso al momento de cada búsqueda).

hiperparametros_finales = {
    "Árbol de Decisión": {"ccp_alpha": arbol_tuned.get_params()["ccp_alpha"]},
    "Random Forest": study_rf.best_params,
    "LightGBM": lgbm_params_finales,
}
tabla_hiperparametros = pd.DataFrame(hiperparametros_finales).T

_log_ml("\n=== HIPERPARÁMETROS FINALES POR MODELO AJUSTADO ===")
_log_ml(
    "(NaN = hiperparámetro no explorado para ese modelo -- cada uno se ajustó con su propio "
    "espacio de búsqueda, ver docs/plan_shap_rule_churn_telecom.md, Fase 4b)"
)
_log_ml(tabla_hiperparametros.to_string())

# %% Tabla comparativa: modelo inicial vs. ajustado (solo los 3 árboles) + análisis de mejoras

nombres_arbol = ["Árbol de Decisión", "Random Forest", "LightGBM"]
tabla_inicial_arboles = tabla_comparativa.loc[nombres_arbol]
tabla_ajustada_arboles = pd.DataFrame(resultados_tuned).set_index("Modelo")
tabla_ajustada_arboles.index = nombres_arbol  # alinea con tabla_inicial_arboles (sin el sufijo "(ajustado)")

comparacion_arboles = pd.concat(
    {"Inicial": tabla_inicial_arboles, "Ajustado": tabla_ajustada_arboles}, axis=1
)

_log_ml("\n=== COMPARACIÓN: MODELO INICIAL vs. AJUSTADO (árboles) ===")
_log_ml(comparacion_arboles.to_string())

_log_ml("\n=== ANÁLISIS DE MEJORAS (ajuste de hiperparámetros) ===")
for nombre in nombres_arbol:
    recall_ini = tabla_inicial_arboles.loc[nombre, "Recall (Churn)"]
    recall_aju = tabla_ajustada_arboles.loc[nombre, "Recall (Churn)"]
    f1_ini = tabla_inicial_arboles.loc[nombre, "F1 (Churn)"]
    f1_aju = tabla_ajustada_arboles.loc[nombre, "F1 (Churn)"]
    auc_ini = tabla_inicial_arboles.loc[nombre, "ROC-AUC"]
    auc_aju = tabla_ajustada_arboles.loc[nombre, "ROC-AUC"]
    brecha_ini = tabla_inicial_arboles.loc[nombre, "Accuracy (train)"] - tabla_inicial_arboles.loc[nombre, "Accuracy (test)"]
    brecha_aju = tabla_ajustada_arboles.loc[nombre, "Accuracy (train)"] - tabla_ajustada_arboles.loc[nombre, "Accuracy (test)"]

    mejora_recall = "mejora" if recall_aju > recall_ini else ("empeora" if recall_aju < recall_ini else "igual")
    mejora_f1 = "mejora" if f1_aju > f1_ini else ("empeora" if f1_aju < f1_ini else "igual")
    mejora_auc = "mejora" if auc_aju > auc_ini else ("empeora" if auc_aju < auc_ini else "igual")
    mejora_brecha = "reduce" if brecha_aju < brecha_ini else ("aumenta" if brecha_aju > brecha_ini else "mantiene")

    _log_ml(
        f"\n{nombre}:\n"
        f"  Recall (Churn):    {recall_ini:.3f} -> {recall_aju:.3f}  ({mejora_recall}, {recall_aju - recall_ini:+.3f})\n"
        f"  F1 (Churn):        {f1_ini:.3f} -> {f1_aju:.3f}  ({mejora_f1}, {f1_aju - f1_ini:+.3f})\n"
        f"  ROC-AUC:           {auc_ini:.3f} -> {auc_aju:.3f}  ({mejora_auc}, {auc_aju - auc_ini:+.3f})\n"
        f"  Brecha train-test: {brecha_ini:.3f} -> {brecha_aju:.3f}  ({mejora_brecha} el sobreajuste, {brecha_aju - brecha_ini:+.3f})"
    )

_log_ml(
    "\nNota: el ajuste optimiza F1 de la clase Churn vía validación cruzada (5-fold) sobre "
    "TRAIN; por eso puede no mejorar todas las métricas simultáneamente sobre TEST (p. ej. "
    "puede subir F1/recall a cambio de un ROC-AUC levemente menor, o reducir la brecha "
    "train-test sacrificando algo de recall) -- es el trade-off esperado al pasar de una "
    "configuración fija a una ajustada específicamente para detectar Churn."
)

# ============================================================================
# PARTE 5: gráfico ROC-AUC comparativo -- Regresión Logística (sin tocar) vs.
# los 3 árboles ajustados en PARTE 4. Paleta categórica de 4 colores vivos en
# orden fijo (no ciclada), validada para daltonismo con el validador de
# paletas del sistema de diseño (CVD ΔE mínimo 25.0, ver
# docs/plan_shap_rule_churn_telecom.md).
# ============================================================================

# %% Curva ROC de cada modelo + línea de referencia "azar" (AUC = 0.5)

COLOR_ROC = {
    "Regresión Logística": "#2a78d6",  # azul
    "Árbol de Decisión (ajustado)": "#eb6834",  # naranja
    "Random Forest (ajustado)": "#1baf7a",  # aqua
    "LightGBM (ajustado)": "#4a3aa7",  # violeta
}


def graficar_roc_comparativo(
    modelos_roc: list[tuple[str, object, pd.DataFrame, pd.Series]], ruta: Path
) -> dict[str, float]:
    """Curva ROC (recall vs. tasa de falsos positivos, en todos los umbrales) de cada modelo + línea de azar."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#898781", label="Azar (AUC = 0.500)")

    auc_por_modelo: dict[str, float] = {}
    for nombre, modelo, X_test_modelo, y_test_modelo in modelos_roc:
        proba = modelo.predict_proba(X_test_modelo)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_modelo, proba)
        auc = roc_auc_score(y_test_modelo, proba)
        auc_por_modelo[nombre] = auc
        ax.plot(fpr, tpr, linewidth=2.5, color=COLOR_ROC[nombre], label=f"{nombre} (AUC = {auc:.3f})")

    ax.set_xlabel("Tasa de Falsos Positivos  (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos  (Recall de Churn)")
    ax.set_title("Curva ROC comparativa: Regresión Logística vs. árboles ajustados")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfico guardado en: {ruta}")
    return auc_por_modelo


ruta_roc = Path(__file__).resolve().parent.parent / "docs" / "roc_auc_comparativa.png"
auc_por_modelo = graficar_roc_comparativo(
    [
        ("Regresión Logística", modelo_logistico, X_test_lineal, y_test_lineal),
        ("Árbol de Decisión (ajustado)", arbol_tuned, X_test_arbol, y_test),
        ("Random Forest (ajustado)", rf_tuned, X_test_arbol, y_test),
        ("LightGBM (ajustado)", lgbm_tuned, X_test_arbol, y_test),
    ],
    ruta_roc,
)

ranking_auc = sorted(auc_por_modelo.items(), key=lambda item: item[1], reverse=True)

_log_ml("\n\n=== GRÁFICO ROC-AUC COMPARATIVO ===")
_log_ml("Gráfico guardado en: docs/roc_auc_comparativa.png")
_log_ml(
    "Qué representa: cada curva grafica, para TODOS los umbrales de decisión posibles (no "
    "solo 0.5, como las métricas de las tablas anteriores), la Tasa de Verdaderos Positivos "
    "(recall: qué % de los churners reales detecta el modelo) contra la Tasa de Falsos "
    "Positivos (qué % de los que NO se van el modelo marca por error como churn). Una curva "
    "más pegada a la esquina superior izquierda logra más recall con menos falsos positivos "
    "en cualquier punto de corte -- por eso el área bajo la curva (AUC) resume en un solo "
    "número qué tan bien separa el modelo ambas clases, sin depender de dónde se fije el "
    "umbral de decisión. La línea diagonal gris (Azar, AUC = 0.500) es la referencia de un "
    "modelo sin poder predictivo (equivalente a tirar una moneda); AUC = 1.000 sería "
    "separación perfecta."
)

_log_ml("\n=== ANÁLISIS ===")
_log_ml(
    "Ranking por AUC: " + " > ".join(f"{nombre} ({auc:.3f})" for nombre, auc in ranking_auc)
)
mejor_nombre, mejor_auc = ranking_auc[0]
peor_nombre, peor_auc = ranking_auc[-1]
_log_ml(
    f"- {mejor_nombre} tiene la curva más cercana a la esquina superior izquierda y el mayor "
    f"AUC ({mejor_auc:.3f}): en el rango de umbrales donde más importa decidir (para no "
    "gastar de más en falsos positivos ni dejar pasar churners), es el que logra más recall "
    "por cada punto de falsos positivos que se tolera."
)
_log_ml(
    f"- {peor_nombre} tiene el AUC más bajo ({peor_auc:.3f}) del grupo, aunque la diferencia "
    f"con el resto es pequeña ({mejor_auc - peor_auc:.3f}) -- las 4 curvas están muy por "
    "encima de la diagonal de azar, así que los 4 modelos separan bien las clases en general; "
    "la brecha entre ellos importa más para elegir el mejor punto de corte que para descartar "
    "a alguno como no apto."
)
_log_ml(
    "- El AUC es agnóstico al umbral, así que no sustituye la tabla de Recall/Precision/F1 en "
    "0.5 (o el umbral de negocio que se elija) -- ambas vistas son complementarias: el AUC dice "
    "qué modelo separa mejor en general, la tabla dice cómo se comporta ese modelo en el punto "
    "de corte que efectivamente se va a usar en producción."
)
