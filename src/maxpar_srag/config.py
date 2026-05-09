"""Configuração central: caminhos, sementes e dicionários do SIVEP-Gripe.

O CSV original do SIVEP-Gripe (Sistema de Informação de Vigilância Epidemiológica
da Gripe / SRAG) usa códigos numéricos para variáveis categóricas. Este módulo
centraliza as traduções oficiais para que toda a pipeline (limpeza, EDA,
modelagem) trabalhe com rótulos legíveis.

**Por quê centralizar:** evita inconsistências entre notebooks, facilita
auditorias e torna ajustes (descobrir um código novo, corrigir uma tradução)
um change único em um único lugar.

Fonte: Dicionário de Dados SIVEP-Gripe (Ministério da Saúde — Secretaria de
Vigilância em Saúde). Cópia local em
`assets/Dicionario_de_Dados_SRAG_SIVEP-Gripe.pdf` (28 páginas, versão jul/2024
— a versão pós-COVID mais recente publicada, cobrindo os campos de vacinação
COVID, vigilância genômica e antígeno adicionados após 2020).

Os mapeamentos abaixo refletem o dicionário vigente para o ano de 2023; alguns
códigos podem ter sido revisados ao longo dos anos (ex.: SUPORT_VEN), portanto
**conferir contra os valores observados na base** durante a FASE 2 (limpeza).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
# PROJECT_ROOT sobe dois níveis a partir deste arquivo:
#   src/maxpar_srag/config.py  →  src/  →  raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# CSV original — só existe até a primeira conversão para Parquet (Fase 0).
# Após a conversão, o CSV é deletado e RAW_DATA_PATH passa a ser o input único
# da pipeline. A constante é mantida para que `data.csv_to_parquet()` continue
# reutilizável em releases futuros do dataset (INFLUD24, INFLUD25, etc.).
RAW_CSV_PATH = PROJECT_ROOT / "assets" / "INFLUD23-23-03-2026.csv"

# Dados brutos em formato Parquet — input principal da pipeline.
# Conteúdo idêntico ao CSV original (todas as colunas como string), apenas
# com formato mais eficiente: ~5-10× menor em disco e ordens de magnitude mais
# rápido para carregar. Casting de tipos acontece em FASE 2 (cleaning).
RAW_DATA_PATH = DATA_RAW / "INFLUD23-23-03-2026.parquet"

# Dados pós-tratamento (Fase 2) — types corretos, missing tratado, derivadas.
CLEAN_PARQUET_PATH = DATA_INTERIM / "srag_2023_clean.parquet"

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Padrão SIVEP-Gripe para variáveis binárias: 1=Sim, 2=Não, 9=Ignorado.
# Cleaning (FASE 2) converterá esses códigos para {1: True, 2: False, 9: NaN}.
# ---------------------------------------------------------------------------
SIM_NAO_IGNORADO = {1: "Sim", 2: "Não", 9: "Ignorado"}
SIM_NAO_BINARY_TO_BOOL = {1: True, 2: False, 9: None}

# Variáveis que seguem o padrão {1, 2, 9}. Lista mantida explicitamente porque
# nem toda coluna binária no CSV tem esse padrão (ex.: SUPORT_VEN tem 3 valores).
# Revisar contra a base durante a FASE 2 antes de aplicar conversão em massa.
BINARY_FIELDS = [
    # Sintomas
    "FEBRE", "TOSSE", "GARGANTA", "DISPNEIA", "DESC_RESP", "SATURACAO",
    "DIARREIA", "VOMITO", "DOR_ABD", "FADIGA", "PERD_OLFT", "PERD_PALA",
    # Comorbidades / fatores de risco
    "PUERPERA", "CARDIOPATI", "HEMATOLOGI", "SIND_DOWN", "HEPATICA", "ASMA",
    "DIABETES", "NEUROLOGIC", "PNEUMOPATI", "IMUNODEPRE", "RENAL", "OBESIDADE",
    "OUT_MORBI", "FATOR_RISC", "TABAG",
    # Vacinação e tratamento
    "VACINA", "MAE_VAC", "M_AMAMENTA", "ANTIVIRAL", "VACINA_COV", "TRAT_COV",
    # Hospitalização e exposição
    "HOSPITAL", "UTI", "NOSOCOMIAL", "AVE_SUINO", "HISTO_VGM",
    # Exames / amostras
    "AMOSTRA", "PCR_RESUL",
    # Contexto administrativo
    "SURTO_SG", "TEM_CPF", "ESTRANG",
]

# ---------------------------------------------------------------------------
# Mapeamentos categóricos específicos
# ---------------------------------------------------------------------------
CS_SEXO = {"M": "Masculino", "F": "Feminino", "I": "Ignorado"}

CS_GESTANT = {
    1: "1º Trimestre",
    2: "2º Trimestre",
    3: "3º Trimestre",
    4: "Idade Gestacional Ignorada",
    5: "Não",
    6: "Não se Aplica",
    9: "Ignorado",
}

CS_RACA = {
    1: "Branca",
    2: "Preta",
    3: "Amarela",
    4: "Parda",
    5: "Indígena",
    9: "Ignorado",
}

CS_ESCOL_N = {
    0: "Sem Escolaridade",
    1: "Fundamental 1º ao 5º ano",
    2: "Fundamental 6º ao 9º ano",
    3: "Ensino Médio",
    4: "Ensino Superior",
    5: "Não se Aplica",
    9: "Ignorado",
}

CS_ZONA = {1: "Urbana", 2: "Rural", 3: "Periurbana", 9: "Ignorado"}

# Idade: TP_IDADE define a unidade de medida de NU_IDADE_N
TP_IDADE = {1: "Dia", 2: "Mês", 3: "Ano"}

# Classificação final do caso (alvo etiológico)
CLASSI_FIN = {
    1: "SRAG por Influenza",
    2: "SRAG por outro vírus respiratório",
    3: "SRAG por outro agente etiológico",
    4: "SRAG não especificado",
    5: "SRAG por COVID-19",
}

# Critério de classificação
CRITERIO = {
    1: "Laboratorial",
    2: "Clínico Epidemiológico",
    3: "Clínico",
    4: "Clínico Imagem",
}

# Evolução do caso — TARGET da modelagem
EVOLUCAO = {
    1: "Cura",
    2: "Óbito",
    3: "Óbito por outras causas",
    9: "Ignorado",
}

# Suporte ventilatório
SUPORT_VEN = {
    1: "Sim, invasivo",
    2: "Sim, não invasivo",
    3: "Não",
    9: "Ignorado",
}

# Resultado do RX de tórax
RAIOX_RES = {
    1: "Normal",
    2: "Infiltrado intersticial",
    3: "Consolidação",
    4: "Misto",
    5: "Outro",
    6: "Não realizado",
    9: "Ignorado",
}

TP_AMOSTRA = {
    1: "Secreção de Naso-orofaringe",
    2: "Lavado Broco-alveolar",
    3: "Tecido post-mortem",
    4: "Outra",
    5: "LCR",
    9: "Ignorado",
}

# Tipo de Influenza identificada (PCR)
TP_FLU_PCR = {1: "Influenza A", 2: "Influenza B"}

# Subtipo Influenza A
PCR_FLUASU = {
    1: "Influenza A(H1N1)pdm09",
    2: "Influenza A (H3N2)",
    3: "Influenza A não subtipada",
    4: "Influenza A não subtipável",
    5: "Inconclusivo",
    6: "Outro",
}

# Linhagem Influenza B
PCR_FLUBLI = {
    1: "Victoria",
    2: "Yamagatha",
    3: "Não realizado",
    4: "Inconclusivo",
    5: "Outro",
}

# ---------------------------------------------------------------------------
# Geografia — UF para Região (IBGE)
# ---------------------------------------------------------------------------
UF_TO_REGIAO = {
    # Norte
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    # Nordeste
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    # Centro-Oeste
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    # Sudeste
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    # Sul
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# ---------------------------------------------------------------------------
# Colunas de data — parseadas como datetime na FASE 1
# ---------------------------------------------------------------------------
DATE_COLUMNS = [
    # Notificação e sintomas
    "DT_NOTIFIC", "DT_SIN_PRI", "DT_NASC",
    # Vacinação influenza / mãe
    "DT_UT_DOSE", "DT_VAC_MAE", "DT_DOSEUNI", "DT_1_DOSE", "DT_2_DOSE",
    # Antiviral
    "DT_ANTIVIR",
    # Hospitalização
    "DT_INTERNA", "DT_ENTUTI", "DT_SAIDUTI",
    # Exames
    "DT_RAIOX", "DT_COLETA", "DT_PCR", "DT_TOMO", "DT_RES_AN",
    "DT_CO_SOR", "DT_RES",
    # Desfecho e digitação
    "DT_EVOLUCA", "DT_ENCERRA", "DT_DIGITA",
    # Viagem
    "DT_VGM", "DT_RT_VGM",
    # Vacinação COVID
    "DOSE_1_COV", "DOSE_2_COV", "DOSE_REF", "DOSE_2REF", "DOSE_ADIC", "DOS_RE_BI",
    # Tratamento COVID
    "DT_TRT_COV",
    # Vigilância genômica
    "VG_DTRES",
]

# ---------------------------------------------------------------------------
# Modelagem
# ---------------------------------------------------------------------------
TARGET_COL = "EVOLUCAO"
TARGET_POSITIVE_VALUE = 2  # Óbito por SRAG (codificado como 1 na binária final)
TARGET_NEGATIVE_VALUE = 1  # Cura (codificado como 0)
TARGET_DROP_VALUES = [3, 9]  # Óbito por outras causas e Ignorado: descartar

# Variáveis pós-evolução: vazamento direto. Excluir antes do treino (FASE 4).
LEAKAGE_FIELDS = [
    "DT_EVOLUCA",
    "DT_ENCERRA",
    "CRITERIO",  # critério de classificação depende do desfecho
]

# Variáveis registradas durante a hospitalização (antes do desfecho).
# Mantidas como features clínicas legítimas — decisão documentada na FASE 4.
PRE_OUTCOME_HOSPITAL_FIELDS = ["UTI", "SUPORT_VEN", "RAIOX_RES"]
