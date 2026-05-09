"""Testes unitários para `maxpar_srag.cleaning`.

Cada teste constrói o menor DataFrame possível para verificar uma propriedade
isolada da função sob teste. Não dependem do Parquet bruto — rodam em
milissegundos e são executáveis em qualquer máquina sem ter o dataset.

Rodar com: `uv run --extra dev pytest tests/`
"""

from __future__ import annotations

import pandas as pd
import pytest

from maxpar_srag import cleaning


# ---------------------------------------------------------------------------
# drop_dead_columns
# ---------------------------------------------------------------------------
def test_drop_dead_columns_removes_empty_and_constant():
    df = pd.DataFrame({
        "TABAG": [None, None, None],          # vazia (em EMPTY_COLUMNS)
        "FATOR_RISC": [1, 1, 1],              # constante (em CONSTANT_COLUMNS)
        "ID_PAIS": ["BRASIL", "BRASIL", "BRASIL"],  # quase-constante
        "FEBRE": [1, 2, 9],                   # informativa, mantém
    })
    out = cleaning.drop_dead_columns(df)
    assert "TABAG" not in out.columns
    assert "FATOR_RISC" not in out.columns
    assert "ID_PAIS" in out.columns           # default mantém quase-constantes
    assert "FEBRE" in out.columns


def test_drop_dead_columns_with_quasi_constant_flag():
    df = pd.DataFrame({"ID_PAIS": ["BRASIL"] * 3, "FEBRE": [1, 2, 9]})
    out = cleaning.drop_dead_columns(df, drop_quasi_constant=True)
    assert "ID_PAIS" not in out.columns
    assert "FEBRE" in out.columns


def test_drop_dead_columns_tolerates_missing_columns():
    """Não levanta se uma coluna esperada não estiver na base."""
    df = pd.DataFrame({"FEBRE": [1, 2, 9]})  # nenhuma das EMPTY/CONSTANT
    out = cleaning.drop_dead_columns(df)
    assert list(out.columns) == ["FEBRE"]


# ---------------------------------------------------------------------------
# filter_temporal
# ---------------------------------------------------------------------------
def test_filter_temporal_keeps_only_target_year():
    df = pd.DataFrame({
        "DT_NOTIFIC": pd.to_datetime(
            ["2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01"]
        )
    })
    out = cleaning.filter_temporal(df, year=2023)
    assert len(out) == 2
    assert out["DT_NOTIFIC"].dt.year.eq(2023).all()


def test_filter_temporal_raises_if_not_datetime():
    df = pd.DataFrame({"DT_NOTIFIC": ["2023-01-01", "2023-02-01"]})
    with pytest.raises(TypeError, match="datetime"):
        cleaning.filter_temporal(df)


def test_filter_temporal_raises_if_column_missing():
    df = pd.DataFrame({"OUTRA": [1, 2]})
    with pytest.raises(KeyError):
        cleaning.filter_temporal(df)


# ---------------------------------------------------------------------------
# filter_modeling_target
# ---------------------------------------------------------------------------
def test_filter_modeling_target_keeps_only_1_and_2():
    df = pd.DataFrame({"EVOLUCAO": ["1", "2", "3", "9", None, "1"]})
    out = cleaning.filter_modeling_target(df)
    assert len(out) == 3
    target = pd.to_numeric(out["EVOLUCAO"], errors="coerce")
    assert set(target.unique()) == {1, 2}


# ---------------------------------------------------------------------------
# cast_dtypes
# ---------------------------------------------------------------------------
def test_cast_dtypes_converts_pure_numeric_strings():
    df = pd.DataFrame({"X": ["1", "2", "3"], "Y": ["a", "b", "c"]})
    out = cleaning.cast_dtypes(df)
    assert pd.api.types.is_numeric_dtype(out["X"])
    # Y permanece textual porque tem valor não-numérico.
    assert not pd.api.types.is_numeric_dtype(out["Y"])


def test_cast_dtypes_preserves_columns_with_mixed_content():
    df = pd.DataFrame({"COD": ["1", "2", "ABC"]})
    out = cleaning.cast_dtypes(df)
    # ABC não converte → coluna preservada como está (não destrói dado).
    assert (out["COD"].astype(str) == ["1", "2", "ABC"]).all()


def test_cast_dtypes_skips_all_null_columns():
    df = pd.DataFrame({"VAZIA": pd.Series([None, None, None], dtype="object")})
    out = cleaning.cast_dtypes(df)
    assert out["VAZIA"].isna().all()


# ---------------------------------------------------------------------------
# parse_dates
# ---------------------------------------------------------------------------
def test_parse_dates_converts_dd_mm_yyyy():
    df = pd.DataFrame({"DT_NOTIFIC": ["01/01/2023", "31/12/2023"]})
    out = cleaning.parse_dates(df, date_columns=["DT_NOTIFIC"])
    assert pd.api.types.is_datetime64_any_dtype(out["DT_NOTIFIC"])
    assert out["DT_NOTIFIC"].iloc[0] == pd.Timestamp("2023-01-01")


def test_parse_dates_is_idempotent_on_already_datetime():
    df = pd.DataFrame({"DT_NOTIFIC": pd.to_datetime(["2023-01-01"])})
    out = cleaning.parse_dates(df, date_columns=["DT_NOTIFIC"])
    assert out["DT_NOTIFIC"].iloc[0] == pd.Timestamp("2023-01-01")


def test_parse_dates_skips_missing_columns():
    df = pd.DataFrame({"X": [1, 2]})
    out = cleaning.parse_dates(df, date_columns=["DT_NAO_EXISTE"])
    assert list(out.columns) == ["X"]


def test_parse_dates_invalid_becomes_nat():
    df = pd.DataFrame({"DT_NOTIFIC": ["invalido", "01/01/2023"]})
    out = cleaning.parse_dates(df, date_columns=["DT_NOTIFIC"])
    assert pd.isna(out["DT_NOTIFIC"].iloc[0])
    assert out["DT_NOTIFIC"].iloc[1] == pd.Timestamp("2023-01-01")


# ---------------------------------------------------------------------------
# decode_binary_fields
# ---------------------------------------------------------------------------
def test_decode_binary_fields_maps_correctly():
    df = pd.DataFrame({"FEBRE": ["1", "2", "9", None]})
    out = cleaning.decode_binary_fields(df, fields=["FEBRE"])
    assert out["FEBRE"].iloc[0] == 1
    assert out["FEBRE"].iloc[1] == 0
    assert pd.isna(out["FEBRE"].iloc[2])  # 9 = ignorado → NaN
    assert pd.isna(out["FEBRE"].iloc[3])  # NaN preservado
    assert str(out["FEBRE"].dtype) == "Float64"


def test_decode_binary_fields_skips_missing_columns():
    df = pd.DataFrame({"OUTRO": ["1", "2"]})
    out = cleaning.decode_binary_fields(df, fields=["FEBRE", "OUTRO"])
    assert "OUTRO" in out.columns
    assert pd.api.types.is_numeric_dtype(out["OUTRO"])


def test_decode_binary_fields_unknown_codes_become_nan():
    """Códigos fora de {1,2,9} viram NaN — defensivo contra dados sujos."""
    df = pd.DataFrame({"FEBRE": ["1", "5", "99"]})
    out = cleaning.decode_binary_fields(df, fields=["FEBRE"])
    assert out["FEBRE"].iloc[0] == 1
    assert pd.isna(out["FEBRE"].iloc[1])
    assert pd.isna(out["FEBRE"].iloc[2])


# ---------------------------------------------------------------------------
# decode_categorical
# ---------------------------------------------------------------------------
def test_decode_categorical_creates_label_column():
    df = pd.DataFrame({"EVOLUCAO": ["1", "2", "3", "9"]})
    out = cleaning.decode_categorical(df)
    assert "EVOLUCAO_LABEL" in out.columns
    assert "EVOLUCAO" in out.columns          # código numérico preservado
    labels = out["EVOLUCAO_LABEL"].tolist()
    assert "Cura" in labels
    assert "Óbito" in labels


def test_decode_categorical_handles_string_keyed_mapping():
    df = pd.DataFrame({"CS_SEXO": ["M", "F", "I"]})
    out = cleaning.decode_categorical(df)
    assert out["CS_SEXO_LABEL"].tolist() == ["Masculino", "Feminino", "Ignorado"]


def test_decode_categorical_unknown_code_becomes_nan_label():
    df = pd.DataFrame({"EVOLUCAO": ["1", "99"]})
    out = cleaning.decode_categorical(df)
    assert out["EVOLUCAO_LABEL"].iloc[0] == "Cura"
    assert pd.isna(out["EVOLUCAO_LABEL"].iloc[1])


# ---------------------------------------------------------------------------
# unify_age
# ---------------------------------------------------------------------------
def test_unify_age_handles_three_units():
    df = pd.DataFrame({
        "NU_IDADE_N": ["60", "12", "365"],
        "TP_IDADE":   ["3",  "2",  "1"],     # anos, meses, dias
    })
    out = cleaning.unify_age(df)
    assert out["IDADE_ANOS"].iloc[0] == pytest.approx(60.0)
    assert out["IDADE_ANOS"].iloc[1] == pytest.approx(1.0)
    assert out["IDADE_ANOS"].iloc[2] == pytest.approx(365 / 365.25, rel=1e-3)


def test_unify_age_clips_invalid_values():
    df = pd.DataFrame({
        "NU_IDADE_N": ["-5", "999", "50"],
        "TP_IDADE":   ["3",  "3",   "3"],
    })
    out = cleaning.unify_age(df)
    assert pd.isna(out["IDADE_ANOS"].iloc[0])  # negativo
    assert pd.isna(out["IDADE_ANOS"].iloc[1])  # >120
    assert out["IDADE_ANOS"].iloc[2] == 50.0


def test_unify_age_unknown_unit_becomes_nan():
    df = pd.DataFrame({"NU_IDADE_N": ["50"], "TP_IDADE": ["7"]})
    out = cleaning.unify_age(df)
    assert pd.isna(out["IDADE_ANOS"].iloc[0])


# ---------------------------------------------------------------------------
# add_derived
# ---------------------------------------------------------------------------
def test_add_derived_creates_faixa_etaria():
    df = pd.DataFrame({"IDADE_ANOS": [2.0, 30.0, 75.0, 90.0]})
    out = cleaning.add_derived(df)
    expected = ["0-4", "20-39", "70-79", "80+"]
    assert out["FAIXA_ETARIA"].astype(str).tolist() == expected


def test_add_derived_maps_uf_to_region():
    df = pd.DataFrame({"SG_UF": ["SP", "BA", "AM", "ZZ"]})
    out = cleaning.add_derived(df)
    assert out["REGIAO_BR"].tolist()[:3] == ["Sudeste", "Nordeste", "Norte"]
    assert pd.isna(out["REGIAO_BR"].iloc[3])  # UF inválida → NaN


def test_add_derived_uses_fallback_uf():
    """Quando SG_UF está vazio, usa SG_UF_NOT."""
    df = pd.DataFrame({
        "SG_UF":     [None, "SP"],
        "SG_UF_NOT": ["RJ", "MG"],
    })
    out = cleaning.add_derived(df)
    assert out["REGIAO_BR"].tolist() == ["Sudeste", "Sudeste"]


def test_add_derived_computes_temporal_intervals():
    df = pd.DataFrame({
        "DT_NOTIFIC": pd.to_datetime(["2023-01-10", "2023-02-15"]),
        "DT_SIN_PRI": pd.to_datetime(["2023-01-05", "2023-02-10"]),
        "DT_INTERNA": pd.to_datetime(["2023-01-12", "2023-02-16"]),
        "DT_EVOLUCA": pd.to_datetime(["2023-01-20", "2023-02-25"]),
    })
    out = cleaning.add_derived(df)
    assert out["TEMPO_SINTOMA_NOTIF"].tolist() == [5, 5]
    assert out["TEMPO_INTERNACAO"].tolist() == [8, 9]


# ---------------------------------------------------------------------------
# clean_pipeline — integração
# ---------------------------------------------------------------------------
def test_clean_pipeline_end_to_end():
    df = pd.DataFrame({
        "DT_NOTIFIC": ["01/03/2023", "15/06/2023", "01/01/2022"],  # 1 fora de 2023
        "DT_SIN_PRI": ["28/02/2023", "10/06/2023", "28/12/2021"],
        "DT_INTERNA": ["02/03/2023", "16/06/2023", "02/01/2022"],
        "DT_EVOLUCA": ["10/03/2023", "25/06/2023", "10/01/2022"],
        "EVOLUCAO":   ["1",          "2",          "1"],
        "NU_IDADE_N": ["35",         "70",         "55"],
        "TP_IDADE":   ["3",          "3",          "3"],
        "SG_UF":      ["SP",         "RJ",         "BA"],
        "FEBRE":      ["1",          "2",          "1"],
        "TABAG":      [None,         None,         None],   # vai sair
        "FATOR_RISC": ["1",          "1",          "1"],    # vai sair
    })
    out = cleaning.clean_pipeline(df, filter_target=True)

    # 1 linha removida pelo filtro temporal; nenhuma pelo target.
    assert len(out) == 2
    # Colunas mortas sumiram.
    assert "TABAG" not in out.columns
    assert "FATOR_RISC" not in out.columns
    # Derivadas e labels presentes.
    for col in ["IDADE_ANOS", "FAIXA_ETARIA", "REGIAO_BR",
                "TEMPO_SINTOMA_NOTIF", "TEMPO_INTERNACAO",
                "EVOLUCAO_LABEL"]:
        assert col in out.columns
    # Binárias decodificadas.
    assert out["FEBRE"].iloc[0] == 1.0
    assert out["FEBRE"].iloc[1] == 0.0
    # Datas parseadas.
    assert pd.api.types.is_datetime64_any_dtype(out["DT_NOTIFIC"])


def test_clean_pipeline_does_not_mutate_input():
    df = pd.DataFrame({
        "DT_NOTIFIC": ["01/03/2023"],
        "EVOLUCAO": ["1"],
        "NU_IDADE_N": ["50"],
        "TP_IDADE": ["3"],
        "SG_UF": ["SP"],
    })
    df_copy = df.copy(deep=True)
    cleaning.clean_pipeline(df)
    pd.testing.assert_frame_equal(df, df_copy)
