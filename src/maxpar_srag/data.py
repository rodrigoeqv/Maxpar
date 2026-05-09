"""Loading e conversão de formato para o dataset SIVEP-Gripe.

Este módulo fronteira entre os formatos de entrada (CSV original) e a
representação interna do projeto (Parquet). Após a conversão inicial, todo o
restante da pipeline lê Parquet exclusivamente — ver `RAW_DATA_PATH` em
`config.py`.

**Por quê uma camada dedicada:** isola a decisão de formato (CSV vs Parquet vs
banco) do resto da pipeline. Se receber INFLUD24 ano que vem, basta chamar
`csv_to_parquet()` novamente — a EDA, o tratamento e a modelagem permanecem
intactos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from . import config


def csv_to_parquet(
    csv_path: Path | str = config.RAW_CSV_PATH,
    parquet_path: Path | str = config.RAW_DATA_PATH,
    *,
    delete_csv: bool = False,
    verbose: bool = True,
) -> int:
    """Converte um CSV SIVEP-Gripe em Parquet preservando valores como string.

    **Por quê polars + streaming:** o CSV SRAG 2023 tem ~300 MB / ~1.6M linhas /
    184 colunas. O scanner lazy do polars lê em batches e escreve Parquet sem
    nunca segurar o arquivo inteiro em RAM (~1-2 GB de pico em pandas eager).

    **Por quê preservar como string:** SIVEP-Gripe codifica booleanas como
    {1, 2, 9} e categóricas multi-classe também como pequenos inteiros. Deixar
    a inferência de tipos adivinhar perderia silenciosamente a distinção entre
    `2 = "Não"` e `9 = "Ignorado"` que é crítica no tratamento. Adiamos todo o
    casting para a FASE 2 (cleaning), onde é explícito e documentado.

    Retorna o número de linhas do Parquet resultante (sanity-check).
    """
    csv_path = Path(csv_path)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    if verbose:
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"Lendo CSV: {csv_path.name} ({size_mb:.1f} MB)")

    # Streaming scan + sink. infer_schema_length=0 mantém todas as colunas Utf8.
    lf = pl.scan_csv(
        csv_path,
        separator=";",
        quote_char='"',
        null_values=["", "NA", "NULL"],
        infer_schema_length=0,  # tudo Utf8 — casting fica para FASE 2
        encoding="utf8",
    )
    lf.sink_parquet(parquet_path, compression="zstd")

    # Verifica relendo o Parquet (também serve como prova de que o arquivo é
    # legível antes de qualquer deleção do CSV original).
    n_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()

    if verbose:
        in_mb = csv_path.stat().st_size / (1024 * 1024)
        out_mb = parquet_path.stat().st_size / (1024 * 1024)
        ratio = in_mb / out_mb if out_mb > 0 else float("inf")
        print(
            f"Parquet escrito: {parquet_path.name} "
            f"({out_mb:.1f} MB, {ratio:.1f}× compressão)"
        )
        print(f"Linhas: {n_rows:,}")

    # Só deleta o CSV depois que o Parquet foi escrito E lido com sucesso.
    if delete_csv:
        csv_path.unlink()
        if verbose:
            print(f"CSV original deletado: {csv_path}")

    return n_rows


def load_raw(
    parquet_path: Path | str = config.RAW_DATA_PATH,
    *,
    engine: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Carrega o Parquet bruto (todas as colunas ainda como string).

    **Por quê pandas como default:** o ecossistema downstream do projeto
    (scikit-learn, lightgbm, xgboost, shap, missingno, seaborn) é todo
    pandas-first. Manter o default em pandas evita `.to_pandas()` repetido em
    cada notebook. polars continua disponível via `engine="polars"` para
    operações que se beneficiam dele (e foi como a conversão CSV→Parquet
    original foi feita).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet bruto não encontrado em {parquet_path}. "
            "Rode `csv_to_parquet()` primeiro se ainda houver o CSV."
        )
    if engine == "pandas":
        return pd.read_parquet(parquet_path)
    if engine == "polars":
        return pl.read_parquet(parquet_path)
    raise ValueError(f"engine deve ser 'pandas' ou 'polars', recebido: {engine!r}")
