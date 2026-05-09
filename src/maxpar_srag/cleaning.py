"""Tratamento da base SRAG 2023 — funções modulares aplicadas pelo notebook 02.

Cada função recebe um DataFrame e devolve uma **cópia** modificada. Nenhuma
muta in-place — facilita compor o pipeline em qualquer ordem nos notebooks e
escrever testes determinísticos.

**Escopo da Fase 2:** estrutura. Drops de colunas mortas, filtro temporal,
casting de dtypes, decodificação de códigos SIVEP-Gripe e variáveis derivadas
de baixo custo (idade, região, tempos). **Imputação de missing fica para a
Fase 4** (feature engineering, dentro de pipeline ajustável apenas em treino).

Listas de drop e flags vêm do profiling da Fase 1 (notebook 01); ver
`STATUS.md` → "Pendências e questões abertas" e a seção "Conclusões" do
notebook 01 para o contexto que motivou cada decisão.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Listas observadas na Fase 1 (notebook 01)
# ---------------------------------------------------------------------------

# 7 colunas 100% vazias na base 2023 — drop sem perda.
EMPTY_COLUMNS: list[str] = [
    "TABAG", "DT_RT_VGM", "PAIS_VGM", "LO_PS_VGM",
    "VG_REINF", "DT_VGM", "CO_PS_VGM",
]

# 21 colunas com 1 único valor (constantes) — drop sem perda de informação.
# `FATOR_RISC=1` é redundante (os campos individuais carregam o sinal).
# Os blocos PCR_*/AN_* são todos zeros porque o agente não foi pesquisado.
CONSTANT_COLUMNS: list[str] = [
    "HISTO_VGM", "FATOR_RISC", "REINF",
    "PCR_PARA1", "PCR_PARA2", "PCR_PARA3", "PCR_PARA4",
    "PCR_VSR", "PCR_RINO", "PCR_METAP", "PCR_ADENO", "PCR_BOCA",
    "PCR_OUTRO", "PCR_SARS2",
    "AN_VSR", "AN_OUTRO", "AN_PARA1", "AN_PARA2", "AN_PARA3",
    "AN_ADENO", "AN_SARS2",
]

# 5 colunas >99% concentradas em um único valor — drop opcional
# (controlado por `drop_quasi_constant` em `drop_dead_columns`).
QUASI_CONSTANT_COLUMNS: list[str] = [
    "ID_PAIS", "CO_PAIS", "POV_CT", "ESTRANG", "FAB_RE_BI",
]


# ---------------------------------------------------------------------------
# Drops estruturais
# ---------------------------------------------------------------------------
def drop_dead_columns(
    df: pd.DataFrame,
    *,
    drop_quasi_constant: bool = False,
) -> pd.DataFrame:
    """Remove colunas vazias e constantes identificadas no profiling.

    Por default mantém as 5 quase-constantes — elas podem virar binárias
    úteis (ex.: `ESTRANG` indica viajante) na Fase 4. Drop com
    `drop_quasi_constant=True` quando se quer um schema mais enxuto.
    """
    to_drop = list(EMPTY_COLUMNS) + list(CONSTANT_COLUMNS)
    if drop_quasi_constant:
        to_drop += list(QUASI_CONSTANT_COLUMNS)

    # Robusto a colunas ausentes (caso a base venha de um dump diferente).
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Filtros de linhas
# ---------------------------------------------------------------------------
def filter_temporal(
    df: pd.DataFrame,
    *,
    year: int = 2023,
    date_col: str = "DT_NOTIFIC",
) -> pd.DataFrame:
    """Mantém apenas casos com `DT_NOTIFIC` no ano informado.

    A base bruta inclui 3.114 casos fora de 2023 (até 2025-06). O escopo do
    desafio é 2023 — descartar é o caminho certo. A coluna precisa estar
    parseada como datetime; chame `parse_dates()` antes.
    """
    if date_col not in df.columns:
        raise KeyError(f"Coluna {date_col!r} não está no DataFrame")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError(
            f"{date_col!r} precisa estar parseado como datetime — "
            "chame parse_dates() antes de filter_temporal()."
        )
    mask = df[date_col].dt.year == year
    return df.loc[mask].copy()


def filter_modeling_target(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas casos com `EVOLUCAO ∈ {1=Cura, 2=Óbito SRAG}`.

    Descarta `3` (óbito por outras causas — ruído de target), `9` (ignorado)
    e NaN. Use **somente** para construir o dataset de modelagem; para EDA
    descritiva (Fase 3), prefira o dataset completo pós-tratamento.
    """
    target = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
    mask = target.isin(
        [config.TARGET_NEGATIVE_VALUE, config.TARGET_POSITIVE_VALUE]
    )
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Casting e parsing
# ---------------------------------------------------------------------------
def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas `object` para numérico onde apropriado.

    A Fase 0 escreveu o Parquet com `infer_schema_length=0` (tudo Utf8); o
    pyarrow promove parte para `datetime64` ao reler, mas ~21 colunas chegam
    como `object` (Decimal/string numérica). `pd.to_numeric(errors="coerce")`
    é robusto para ambos os casos — vira NaN o que não converte.

    Só casteia colunas onde **todos os valores não-nulos** parsearem com
    sucesso. Isso evita destruir silenciosamente colunas que de fato são
    strings (ex.: códigos alfanuméricos).
    """
    out = df.copy()
    candidate_cols = [
        c for c in out.columns
        if pd.api.types.is_object_dtype(out[c])
        or pd.api.types.is_string_dtype(out[c])
    ]
    for col in candidate_cols:
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        coerced = pd.to_numeric(non_null, errors="coerce")
        # Só substitui se NENHUM valor não-nulo virou NaN no cast.
        if coerced.isna().sum() == 0:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_dates(
    df: pd.DataFrame,
    *,
    date_columns: Iterable[str] = config.DATE_COLUMNS,
    date_format: str = "%d/%m/%Y",
) -> pd.DataFrame:
    """Parseia colunas-data ainda em string (formato dd/mm/aaaa).

    Várias colunas de `DATE_COLUMNS` já vieram como `datetime64` na escrita
    do Parquet; outras (ex.: `DT_NOTIFIC`) ainda estão como string. Esta
    função é idempotente — pula colunas que já são datetime ou que não
    existem no DataFrame.
    """
    out = df.copy()
    for col in date_columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        out[col] = pd.to_datetime(
            out[col], format=date_format, errors="coerce"
        )
    return out


# ---------------------------------------------------------------------------
# Decodificação SIVEP-Gripe
# ---------------------------------------------------------------------------
def decode_binary_fields(
    df: pd.DataFrame,
    *,
    fields: Iterable[str] = config.BINARY_FIELDS,
) -> pd.DataFrame:
    """Converte campos binários `{1=Sim, 2=Não, 9=Ignorado}` → `{1, 0, NaN}`.

    Trata NaN explícito como NaN (não como 0). Pula colunas ausentes —
    `BINARY_FIELDS` no config inclui campos como `TABAG` e `HISTO_VGM` que
    são dropados pelo `drop_dead_columns` antes desta função no pipeline.

    **Cuidado:** `SUPORT_VEN` tem 3 valores válidos (1=invasivo, 2=não
    invasivo, 3=não) e **não** está em `BINARY_FIELDS`. Vai pelo
    `decode_categorical()`.
    """
    out = df.copy()
    mapping = {1: 1, 2: 0, 9: np.nan}
    for col in fields:
        if col not in out.columns:
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric.map(mapping).astype("Float64")
    return out


# Mapas categóricos do config a aplicar em colunas existentes. A função
# `decode_categorical()` cria uma versão `<COL>_LABEL` legível e mantém o
# código numérico original (útil para modelos que esperam ints).
DEFAULT_CATEGORICAL_MAPS: dict[str, dict] = {
    "CS_SEXO": config.CS_SEXO,
    "CS_GESTANT": config.CS_GESTANT,
    "CS_RACA": config.CS_RACA,
    "CS_ESCOL_N": config.CS_ESCOL_N,
    "CS_ZONA": config.CS_ZONA,
    "CLASSI_FIN": config.CLASSI_FIN,
    "CRITERIO": config.CRITERIO,
    "EVOLUCAO": config.EVOLUCAO,
    "SUPORT_VEN": config.SUPORT_VEN,
    "RAIOX_RES": config.RAIOX_RES,
    "TP_AMOSTRA": config.TP_AMOSTRA,
    "TP_FLU_PCR": config.TP_FLU_PCR,
    "PCR_FLUASU": config.PCR_FLUASU,
    "PCR_FLUBLI": config.PCR_FLUBLI,
}


def decode_categorical(
    df: pd.DataFrame,
    *,
    mappings: dict[str, dict] = DEFAULT_CATEGORICAL_MAPS,
) -> pd.DataFrame:
    """Cria colunas `<COL>_LABEL` com rótulos legíveis preservando o código.

    Modelos consomem o código numérico (`CLASSI_FIN`); humanos e gráficos
    consomem o rótulo (`CLASSI_FIN_LABEL`). Manter os dois evita bugs do
    tipo "perdi o código quando virei tudo string para o gráfico".

    Códigos não mapeados viram `NaN` no `_LABEL` — sinal de que o dicionário
    está desatualizado; vale conferir contra o PDF oficial.
    """
    out = df.copy()
    for col, mapping in mappings.items():
        if col not in out.columns:
            continue
        # CS_SEXO usa chaves str; o resto usa int. Detecta automaticamente.
        sample_key = next(iter(mapping))
        if isinstance(sample_key, str):
            source = out[col].astype("string")
        else:
            source = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        out[f"{col}_LABEL"] = source.map(mapping).astype("string")
    return out


# ---------------------------------------------------------------------------
# Idade — unificação de NU_IDADE_N + TP_IDADE
# ---------------------------------------------------------------------------
def unify_age(
    df: pd.DataFrame,
    *,
    value_col: str = "NU_IDADE_N",
    unit_col: str = "TP_IDADE",
    out_col: str = "IDADE_ANOS",
) -> pd.DataFrame:
    """Cria `IDADE_ANOS` a partir de `(NU_IDADE_N, TP_IDADE)`.

    `TP_IDADE` codifica a unidade: 1=dia, 2=mês, 3=ano. Aplicamos a
    conversão preservando frações (relevante para neonatos e lactentes
    quando comparados a adultos no mesmo histograma).

    Valores fora do intervalo `[0, 120]` viram `NaN` — a base SIVEP-Gripe
    tem registros aberrantes (ex.: 999 anos) que são erros de digitação.
    """
    out = df.copy()
    value = pd.to_numeric(out[value_col], errors="coerce")
    unit = pd.to_numeric(out[unit_col], errors="coerce")

    factor = pd.Series(np.nan, index=out.index, dtype="float64")
    factor[unit == 1] = 1 / 365.25  # dias → anos
    factor[unit == 2] = 1 / 12      # meses → anos
    factor[unit == 3] = 1.0          # já em anos

    age_years = value * factor
    age_years = age_years.where((age_years >= 0) & (age_years <= 120))
    out[out_col] = age_years.astype("Float64")
    return out


# ---------------------------------------------------------------------------
# Variáveis derivadas
# ---------------------------------------------------------------------------
# Faixas etárias usadas em boletins do MS para SRAG.
AGE_BINS = [-0.01, 4, 19, 39, 59, 69, 79, 120]
AGE_LABELS = ["0-4", "5-19", "20-39", "40-59", "60-69", "70-79", "80+"]


def add_derived(
    df: pd.DataFrame,
    *,
    uf_col: str = "SG_UF",
    fallback_uf_col: str = "SG_UF_NOT",
) -> pd.DataFrame:
    """Cria variáveis derivadas baratas usadas pela EDA e pela modelagem.

    - `FAIXA_ETARIA` — categórica ordinal de `IDADE_ANOS` (boletins MS).
    - `REGIAO_BR` — UF (residência, fallback para notificação) → região IBGE.
    - `TEMPO_SINTOMA_NOTIF` — dias entre primeiro sintoma e notificação.
      Reflete tempestividade do sistema de vigilância.
    - `TEMPO_INTERNACAO` — dias entre internação e desfecho. **Atenção:**
      depende de `DT_EVOLUCA`, que é vazamento na modelagem (Fase 4 vai
      excluir). Útil para EDA (Fase 3) apenas.

    Requer `IDADE_ANOS` (rode `unify_age()` antes) e datas parseadas
    (`parse_dates()` antes).
    """
    out = df.copy()

    if "IDADE_ANOS" in out.columns:
        out["FAIXA_ETARIA"] = pd.cut(
            out["IDADE_ANOS"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            include_lowest=True,
        )

    # UF preferencial: residência (SG_UF). Se ausente, cai para a UF da
    # unidade notificadora (SG_UF_NOT) — quase sempre coincide.
    uf_source = None
    if uf_col in out.columns:
        uf_source = out[uf_col].astype("string").str.upper()
        if fallback_uf_col in out.columns:
            fallback = out[fallback_uf_col].astype("string").str.upper()
            uf_source = uf_source.fillna(fallback)
    elif fallback_uf_col in out.columns:
        uf_source = out[fallback_uf_col].astype("string").str.upper()

    if uf_source is not None:
        out["REGIAO_BR"] = uf_source.map(config.UF_TO_REGIAO).astype("string")

    if {"DT_NOTIFIC", "DT_SIN_PRI"}.issubset(out.columns):
        delta = (out["DT_NOTIFIC"] - out["DT_SIN_PRI"]).dt.days
        out["TEMPO_SINTOMA_NOTIF"] = delta.astype("Int64")

    if {"DT_EVOLUCA", "DT_INTERNA"}.issubset(out.columns):
        delta = (out["DT_EVOLUCA"] - out["DT_INTERNA"]).dt.days
        out["TEMPO_INTERNACAO"] = delta.astype("Int64")

    return out


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def clean_pipeline(
    df: pd.DataFrame,
    *,
    year: int = 2023,
    drop_quasi_constant: bool = False,
    filter_target: bool = False,
) -> pd.DataFrame:
    """Aplica o tratamento completo da Fase 2 em ordem segura.

    Por default **não** filtra pelo target — para EDA (Fase 3) você quer
    todos os casos. Passe `filter_target=True` para gerar o dataset de
    modelagem (Fase 4 em diante).

    Ordem importa:
      1. Drop colunas mortas (reduz custo das próximas operações)
      2. Cast dtypes (object → numeric onde possível)
      3. Parse datas (necessário para o filtro temporal)
      4. Filtro temporal (só 2023)
      5. Unificação de idade (precisa de NU_IDADE_N/TP_IDADE numéricos)
      6. Decode binárias e categóricas
      7. Variáveis derivadas (precisam de IDADE_ANOS e datas)
      8. (Opcional) filtro do target para o dataset de modelagem
    """
    df = drop_dead_columns(df, drop_quasi_constant=drop_quasi_constant)
    df = cast_dtypes(df)
    df = parse_dates(df)
    df = filter_temporal(df, year=year)
    df = unify_age(df)
    df = decode_binary_fields(df)
    df = decode_categorical(df)
    df = add_derived(df)
    if filter_target:
        df = filter_modeling_target(df)
    return df
