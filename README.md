# Análise SRAG 2023

Análise da base de **Síndrome Respiratória Aguda Grave (SRAG) 2023** do Open Data SUS (SIVEP-Gripe).

---

## Sumário executivo

A pergunta que orienta o trabalho é direta: dado um caso de SRAG notificado em 2023, qual a probabilidade de óbito? E o que os dados dizem sobre prioridades de saúde pública no Brasil?

Em uma linha, o ano de 2023 foi marcado por SRAG pediátrica em volume e SRAG idosa em letalidade. O modelo final (XGBoost com hiperparâmetros otimizados) alcança **AUC-PR de 0,5531 e AUC-ROC de 0,9055** no teste temporal de julho a dezembro de 2023, aproximadamente 5,5 vezes acima do baseline aleatório. O modelo identifica corretamente os óbitos quando há sinais clínicos de gravidade, como ventilação invasiva, internação em UTI e idade avançada.

### Achados-chave

| # | Achado | Magnitude |
|---|---|---|
| 1 | Pico de SRAG em maio/2023 (SE 21) foi pediátrico, não COVID | 42% dos casos em menores de 5 anos |
| 2 | Gradiente etário monotônico na letalidade | 1,2% (0-4) a 27,7% (80+) — variação de 23 vezes |
| 3 | Paradoxo de Simpson na vacinação COVID (efeito real é protetor após estratificar por idade) | Em 80+: vacinados com queda de 10,1pp em relação a não vacinados |
| 4 | Suporte ventilatório invasivo é o melhor preditor clínico | Letalidade de 40,95% (invasivo) contra 4,07% (sem suporte) |
| 5 | Comorbidades raras e graves têm efeito mais limpo que as comuns | Hepática 30%, renal 27%, imunodepressão 24% (sobrevivem à estratificação por idade) |

### Modelo final

| Métrica | Valor | Leitura |
|---|---|---|
| AUC-PR (primária) | 0,5531 | 5,5 vezes o baseline aleatório (cerca de 0,10) |
| AUC-ROC | 0,9055 | Discriminação elevada |
| Brier Score | 0,0915 | Calibração imperfeita; recalibração recomendada antes de uso clínico |
| Threshold operacional | 0,319 (Youden) ou 0,515 (Recall ≥ 70%) | Escolha depende do uso: triagem ou alerta |

Top features pelo SHAP: `SUPORT_VEN_ORD` > `UTI` > `IDADE_ANOS` — coerente com a prática clínica.

### Recomendações de saúde pública (detalhadas no notebook 08)

1. Reforço da capacidade pediátrica (UTI e leitos clínicos infantis) antes do pico anual de VSR e Influenza.
2. Priorização vacinal por perfil idade × comorbidade, com busca ativa de idosos 70+ com menos de quatro doses.
3. Sistema de alerta antecipado, com pipeline semanal alimentando painel público nos moldes do InfoGripe (Fiocruz).
4. Padronização da qualidade do registro SIVEP-Gripe, com auditoria amostral mensal por UF.
5. Pesquisa direcionada em `CLASSI_FIN=3` (SRAG por outro agente) em idosos, dada a letalidade desproporcional sem agente identificado.

---

## Como reproduzir

### Pré-requisitos

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (gerenciador de dependências)
- O arquivo `INFLUD23-23-03-2026.parquet` em `data/raw/` (não versionado, 22 MB)

### Setup

```powershell
uv sync                 # cria .venv e instala dependências
uv run jupyter lab      # abre o JupyterLab
```

Executar os notebooks `01` a `08` em ordem. O `random_state` fica fixo em [src/maxpar_srag/config.py](src/maxpar_srag/config.py) (valor 42).

### Testes

```powershell
uv run pytest           # 29 testes em tests/test_cleaning.py
```

---

## Navegação dos notebooks

| Notebook | Fase | Conteúdo |
|---|---|---|
| [01_exploracao_inicial.ipynb](notebooks/01_exploracao_inicial.ipynb) | Profiling | Validação das 194 colunas contra o dicionário SIVEP-Gripe; range temporal e missing global perfilados |
| [02_tratamento_dados.ipynb](notebooks/02_tratamento_dados.ipynb) | Limpeza | Dataset limpo com 276.339 linhas e 185 colunas; cada decisão de pré-processamento documentada em markdown |
| [03_eda_descritiva.ipynb](notebooks/03_eda_descritiva.ipynb) | EDA | 19 figuras cobrindo as nove dimensões do enunciado; achados-chave (paradoxo de Simpson, gradiente etário, confundimento) |
| [04_feature_engineering.ipynb](notebooks/04_feature_engineering.ipynb) | Features | Split temporal de 145k em treino e 103k em teste; 45 features; indicadores `_MISS` |
| [05_modelagem_baseline.ipynb](notebooks/05_modelagem_baseline.ipynb) | Modelagem | Oito configurações (Dummy a XGBoost) com CV temporal; Optuna para tuning; XGBoost com tuning como melhor escolha |
| [07_avaliacao_e_interpretacao.ipynb](notebooks/07_avaliacao_e_interpretacao.ipynb) | Avaliação | Métricas finais, thresholds operacionais, SHAP, análise de equidade |
| [08_conclusoes.ipynb](notebooks/08_conclusoes.ipynb) | Síntese | Síntese narrativa, principais insights e fatores, recomendações e limitações |

---

## Estrutura

```
.
├── assets/
│   ├── Desafio_Célula de Pesquisa.docx                # enunciado original
│   └── Dicionario_de_Dados_SRAG_SIVEP-Gripe.pdf       # MS/SVS, jul/2024 — referência canônica
├── data/
│   ├── raw/                  # parquet bruto (não versionado, 22 MB)
│   ├── interim/              # parquets pós-tratamento
│   └── processed/            # X_train, y_train, X_test, y_test
├── models/
│   └── best_model.joblib     # pipeline sklearn serializado (XGBoost com tuning)
├── notebooks/                # narrativa numerada 01..08
├── reports/
│   ├── figures/              # PNGs gerados pelos notebooks
│   ├── relatorio_tecnico.pdf # relatório técnico
│   └── apresentacao_final.pdf # slides para defesa oral
├── src/maxpar_srag/          # código reutilizável (importado pelos notebooks)
│   ├── config.py             # paths, RANDOM_STATE, dicionários SIVEP-Gripe
│   ├── data.py               # csv_to_parquet, load_raw (engine pandas|polars)
│   ├── cleaning.py           # funções de limpeza + clean_pipeline
│   ├── features.py           # feature engineering, build_preprocessor, temporal_split
│   └── models.py             # builders (LR/RF/XGB/LGBM) + Optuna + plots
├── tests/
│   └── test_cleaning.py      # 29 testes pytest
└── README.md                 # este arquivo
```

---

## Decisões metodológicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Estrutura | Híbrida (notebooks + `src/`) | Notebooks concentram a narrativa; módulos reutilizáveis e testáveis vivem em `src/` |
| Storage | Parquet zstd | Cerca de 14 vezes menor que CSV; carregamento de 10 a 20 vezes mais rápido |
| Target | `EVOLUCAO ∈ {1=Cura, 2=Óbito SRAG}` binário | Pergunta do desafio; descarte de 3 (óbito por outras causas) e 9 (ignorado) por ruído ou incerteza |
| Métrica primária | AUC-PR | Óbito é a classe rara (cerca de 10%); AUC-PR é mais sensível ao desempenho na classe minoritária do que AUC-ROC |
| Split | Temporal (Jan-Jun em treino, Jul-Dez em teste) | Simula uso real (modelo treinado em dados passados); capta o shift de composição etiológica (VSR a COVID) |
| Vazamento | Exclusão de `DT_EVOLUCA`, `DT_ENCERRA` e `CRITERIO`; manutenção de `UTI`, `SUPORT_VEN_ORD` e `RAIOX_RES` | Variáveis hospitalares ocorrem antes do desfecho; uso clínico legítimo |
| Imbalance | `class_weight='balanced'` e `scale_pos_weight=9` | Mais simples e robusto que SMOTE em dados tabulares de alta dimensionalidade |
| Tuning | Optuna TPE (50 a 80 trials, CV 3-fold em subset) | Bayesiano; mais eficiente que grid ou random search |
| Reprodutibilidade | `random_state=42` fixo em `config.py` | Seed única importada por todos os módulos |

---

## Entregáveis

1. **Repositório** (este) — código e notebooks executáveis a partir de um clone limpo.
2. **[Relatório técnico (PDF)](reports/relatorio_tecnico.pdf)** — documento de leitura assíncrona com decisões e figuras-chave.
3. **[Apresentação (PDF)](reports/apresentacao_final.pdf)** — slides para defesa oral.
4. **README** (este arquivo) — sumário executivo e navegação.

---

## Stack

`Python 3.11` · `pandas 3.0` · `polars 1.40` · `scikit-learn 1.8` · `xgboost 3.2` · `lightgbm 4.6` · `optuna` · `shap` · `matplotlib` · `seaborn` · `pytest` · `uv`
