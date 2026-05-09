"""Feature engineering para modelagem SRAG 2023 (Fase 4).

Recebe o dataset limpo (`data/interim/srag_2023_clean.parquet`) e produz
X_train, X_test, y_train, y_test prontos para sklearn/lightgbm/xgboost.

Decisões de vazamento documentadas:
  ❌ Excluídos: DT_EVOLUCA, DT_ENCERRA, CRITERIO, TEMPO_INTERNACAO
  ✅ Mantidos: UTI, SUPORT_VEN, RAIOX_RES (registrados antes do desfecho)
Ver STATUS.md e notebook 04 para a justificativa completa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

# ---------------------------------------------------------------------------
# Constantes de feature groups (usadas no ColumnTransformer)
# ---------------------------------------------------------------------------

# Suporte ventilatório: recodifica 1=invasivo→2, 2=não-invasivo→1, 3=não→0, 9=NaN
# Preserva a ordinalidade clínica (invasivo > não-invasivo > sem suporte).
SUPORT_VEN_MAP = {1: 2, 2: 1, 3: 0, 9: np.nan}

# Escolaridade ordinal: 5="Não se Aplica" e 9="Ignorado" viram NaN
CS_ESCOL_N_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: np.nan, 9: np.nan}

# Colunas de dose COVID — para contagem N_DOSES_REGISTRADAS
DOSE_COV_COLS = [
    "DOSE_1_COV", "DOSE_2_COV", "DOSE_REF", "DOSE_2REF", "DOSE_ADIC", "DOS_RE_BI",
]

# Features por tier (definidas na síntese final do notebook 03 / STATUS.md)
NUMERIC_FEATURES = [
    "IDADE_ANOS",
    "SEM_NOT",
    "TEMPO_SINTOMA_NOTIF",
    "N_DOSES_REGISTRADAS",
]

BINARY_FEATURES = [
    # Cuidado hospitalar (pré-desfecho legítimo)
    "UTI",
    # Vacinação
    "VACINA_COV",
    # Comorbidades Tier 1
    "CARDIOPATI", "DIABETES", "IMUNODEPRE", "RENAL", "HEPATICA",
    # Comorbidades Tier 2/3
    "OBESIDADE", "PNEUMOPATI", "NEUROLOGIC", "ASMA", "HEMATOLOGI",
    "SIND_DOWN", "OUT_MORBI",
    # Sintomas individuais (Tier 3)
    "FEBRE", "TOSSE", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "FADIGA", "PERD_OLFT", "PERD_PALA",
    # Feature composta (criada em build_features)
    "RESP_SEVERIDADE",
]

# Ordinais: já convertidas para escala numérica em build_features
ORDINAL_FEATURES = [
    "SUPORT_VEN_ORD",  # 0=não, 1=não invasivo, 2=invasivo
    "CS_ESCOL_N_ORD",  # 0-4 (ordinal educacional)
]

OHE_FEATURES = [
    "CLASSI_FIN",
    "CS_SEXO",
    "CS_RACA",
    "RAIOX_RES",
    "REGIAO_BR",
    "CS_ZONA",
]

# Comorbidades com ~70% missing: "não preenchido" ≠ "ausente" em fichas SIVEP-Gripe.
# O campo é preenchido quando a comorbidade está presente ou foi ativamente investigada;
# missing pode indicar "não avaliado" — sinal distinto de 0 para o modelo.
MISS_INDICATOR_SOURCES = [
    "CARDIOPATI", "DIABETES", "IMUNODEPRE", "RENAL", "HEPATICA",
    "OBESIDADE", "NEUROLOGIC", "ASMA",
]
MISS_FEATURES = [f"{c}_MISS" for c in MISS_INDICATOR_SOURCES]

# Cap de outlier para TEMPO_SINTOMA_NOTIF.
# Clinicamente, >30 dias entre sintoma e notificação de SRAG é erro de registro
# (a doença tem curso de dias/semanas; internação ocorre em dias, não meses).
# Percentil 99 do treino ≈ 31 dias — alinha com o limite clínico.
TEMPO_SINTOMA_CAP = 30

# Colunas que vazam informação do desfecho — excluir antes do treino.
LEAKAGE_COLS = list(config.LEAKAGE_FIELDS) + [
    "TEMPO_INTERNACAO",  # depende de DT_EVOLUCA
    "EVOLUCAO_LABEL",    # label string do target
]


# ---------------------------------------------------------------------------
# 1. Target
# ---------------------------------------------------------------------------

def make_target(df: pd.DataFrame) -> pd.Series:
    """Binariza EVOLUCAO: 1=Óbito, 0=Cura. Descarta {3, 9, NaN}.

    Retorna Series com índice preservado (alinhado com X após o filtro).
    """
    target = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
    mask = target.isin([config.TARGET_NEGATIVE_VALUE, config.TARGET_POSITIVE_VALUE])
    binary = target[mask].map(
        {config.TARGET_NEGATIVE_VALUE: 0, config.TARGET_POSITIVE_VALUE: 1}
    ).astype("int8")
    return binary


# ---------------------------------------------------------------------------
# 2. Feature construction (pré-pipeline)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformações e cria derivadas *antes* do pipeline sklearn.

    Não imputa missing — isso fica para o ColumnTransformer (ajustado apenas
    em treino, sem data leakage). Retorna subset de colunas relevantes.
    """
    out = df.copy()

    # --- Suporte ventilatório: ordinal clínico ---
    sv = pd.to_numeric(out["SUPORT_VEN"], errors="coerce")
    out["SUPORT_VEN_ORD"] = sv.map(SUPORT_VEN_MAP).astype("Float64")

    # --- Escolaridade ordinal (limpa Não-se-Aplica / Ignorado) ---
    escol = pd.to_numeric(out["CS_ESCOL_N"], errors="coerce")
    out["CS_ESCOL_N_ORD"] = escol.map(CS_ESCOL_N_MAP).astype("Float64")

    # --- Feature composta: severidade respiratória ---
    # 1 se qualquer um dos três sinais = 1; 0 se todos = 0; NaN se tudo missing.
    resp_cols = ["DISPNEIA", "DESC_RESP", "SATURACAO"]
    existing_resp = [c for c in resp_cols if c in out.columns]
    if existing_resp:
        resp_df = out[existing_resp].apply(pd.to_numeric, errors="coerce")
        out["RESP_SEVERIDADE"] = resp_df.max(axis=1).astype("Float64")

    # --- Número de doses COVID registradas ---
    existing_doses = [c for c in DOSE_COV_COLS if c in out.columns]
    if existing_doses:
        # Conta colunas de data não-nulas como proxy de doses recebidas.
        dose_notna = out[existing_doses].notna().sum(axis=1)
        out["N_DOSES_REGISTRADAS"] = dose_notna.astype("Float64")
    else:
        out["N_DOSES_REGISTRADAS"] = np.nan

    # --- Outlier capping: TEMPO_SINTOMA_NOTIF ---
    # Valores >30 dias são erros de registro — SRAG tem curso clínico de dias/semanas.
    # IDADE_ANOS já foi clipada em [0, 120] na Fase 2; não requer cap adicional.
    if "TEMPO_SINTOMA_NOTIF" in out.columns:
        out["TEMPO_SINTOMA_NOTIF"] = pd.to_numeric(
            out["TEMPO_SINTOMA_NOTIF"], errors="coerce"
        ).clip(lower=0, upper=TEMPO_SINTOMA_CAP)

    # --- Indicadores de missing para comorbidades ---
    # Criados ANTES da imputação para preservar o sinal de "não preenchido".
    for src_col in MISS_INDICATOR_SOURCES:
        if src_col in out.columns:
            out[f"{src_col}_MISS"] = (
                pd.to_numeric(out[src_col], errors="coerce").isna().astype("float64")
            )

    # --- CS_SEXO: garantir string maiúscula ---
    if "CS_SEXO" in out.columns:
        out["CS_SEXO"] = out["CS_SEXO"].astype("string").str.upper().str.strip()
        # "I" (ignorado) → NaN para OHE tratar como unknown
        out["CS_SEXO"] = out["CS_SEXO"].where(out["CS_SEXO"].isin(["M", "F"]))

    # --- Selecionar apenas colunas de features ---
    all_feature_cols = (
        NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES + OHE_FEATURES + MISS_FEATURES
    )
    existing_cols = [c for c in all_feature_cols if c in out.columns]
    result = out[existing_cols].copy()

    # Sklearn não aceita pandas nullable types (Float64/Int64 com pd.NA).
    # Converte numérico/binário/ordinal para float64 (pd.NA → np.nan).
    # OHE columns ficam como object (str) — sklearn SimpleImputer aceita.
    ohe_set = set(OHE_FEATURES)
    for col in result.columns:
        if col in ohe_set:
            # Sklearn usa `X != X` para detectar NaN em object arrays.
            # pd.NA levanta TypeError nessa comparação; np.nan (float) funciona
            # porque np.nan != np.nan é True. Por isso substituímos pd.NA por np.nan.
            arr = result[col].astype(object).values.copy()
            arr[pd.isna(result[col]).values] = np.nan
            result[col] = arr
        else:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
    return result


# ---------------------------------------------------------------------------
# 3. Split temporal
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    *,
    date_col: str = "DT_NOTIFIC",
    cutoff_month: int = 6,
    cutoff_year: int = 2023,
) -> tuple[pd.Index, pd.Index]:
    """Retorna (train_idx, test_idx) por corte temporal semestral.

    Treino: Jan–Jun 2023 (1º semestre).
    Teste : Jul–Dez 2023 (2º semestre).

    Por quê split temporal e não aleatório: simula uso real — modelo treinado
    em dados históricos, avaliado em casos futuros. Previne data leakage
    temporal (ex.: modelos que aprendem padrões do 2º semestre contaminando
    o treino). Standard em epidemiologia preditiva.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError(f"{date_col!r} precisa estar como datetime.")

    cutoff = pd.Timestamp(year=cutoff_year, month=cutoff_month, day=30)
    train_idx = df.index[df[date_col] <= cutoff]
    test_idx = df.index[df[date_col] > cutoff]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# 4. Pipeline sklearn (imputation + encoding)
# ---------------------------------------------------------------------------

def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Constrói ColumnTransformer para imputation + encoding.

    Imputa apenas em treino (fit), aplica em teste (transform).
    Estratégia de imputação:
      - Numérico: mediana (robusta a outliers — IDADE_ANOS tem distribuição
        assimétrica; TEMPO_SINTOMA_NOTIF tem caudas longas).
      - Binário/Ordinal: moda (most_frequent) — preserva a classe dominante.
      - OHE categórico: moda + indicador de "missing" como categoria "Unknown".

    Retorna o preprocessor não ajustado — chamar .fit(X_train) no notebook.
    """
    num_cols = [c for c in NUMERIC_FEATURES if c in feature_cols]
    # MISS_FEATURES são binárias (0/1, sem missing por construção) — tratadas como bin.
    bin_cols = [c for c in BINARY_FEATURES + MISS_FEATURES if c in feature_cols]
    ord_cols = [c for c in ORDINAL_FEATURES if c in feature_cols]
    ohe_cols = [c for c in OHE_FEATURES if c in feature_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  # necessário para Regressão Logística (coeficientes comparáveis)
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if bin_cols:
        transformers.append(("bin", bin_pipe, bin_cols))
    if ord_cols:
        transformers.append(("ord", ord_pipe, ord_cols))
    if ohe_cols:
        transformers.append(("ohe", ohe_pipe, ohe_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------------
# 5. Persistência
# ---------------------------------------------------------------------------

def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    out_dir=config.DATA_PROCESSED,
) -> None:
    """Salva os quatro splits em Parquet (zstd) em `data/processed/`."""
    out_dir = out_dir if hasattr(out_dir, "__fspath__") else __import__("pathlib").Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(out_dir / "X_train.parquet", compression="zstd", index=True)
    X_test.to_parquet(out_dir / "X_test.parquet", compression="zstd", index=True)
    y_train.to_frame(name="target").to_parquet(out_dir / "y_train.parquet", compression="zstd", index=True)
    y_test.to_frame(name="target").to_parquet(out_dir / "y_test.parquet", compression="zstd", index=True)
