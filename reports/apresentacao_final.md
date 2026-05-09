---
marp: true
theme: default
paginate: true
size: 16:9
header: 'SRAG 2023 · Rodrigo'
footer: '2026'
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 24px;
    padding: 48px;
  }
  h1 { color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 8px; }
  h2 { color: #2a4d7a; }
  h3 { color: #34568b; }
  table { font-size: 20px; }
  strong { color: #c8102e; }
  blockquote { border-left: 4px solid #1e3a5f; padding-left: 16px; color: #555; }
  section.title { background: #1e3a5f; color: white; text-align: center; }
  section.title h1 { color: white; border: none; }
  section.title h2 { color: #c8c8c8; }
  section.section-divider { background: #2a4d7a; color: white; }
  section.section-divider h1 { color: white; border: none; font-size: 56px; }
---

<!-- _class: title -->

# Análise SRAG 2023

## Predição de Evolução do Caso e Recomendações de Saúde Pública

Rodrigo · *rodrigoeqv@gmail.com*

---

# Roteiro

1. **Problema e dados** — escopo da análise e características da base
2. **Decisões metodológicas** — split temporal, vazamento e métrica
3. **Cinco insights da EDA** — o que os dados revelam
4. **Modelo final** — XGBoost com tuning: AUC-PR 0,55 e AUC-ROC 0,91
5. **Equidade e calibração** — onde o modelo é forte e onde requer cautela
6. **Recomendações de saúde pública** — o passo acionável
7. **Limitações e próximos passos**

> Materiais: 7 fases, 8 notebooks, 38 figuras e 1 modelo serializado.

---

<!-- _class: section-divider -->

# 1. Problema e dados

---

# Escopo da análise

| # | Entrega |
|---|---|
| 1 | **Tratamento de dados** — correção e documentação |
| 2 | **Análise descritiva** — tendências, geografia e fatores de risco |
| 3 | **Modelos de ML** — predição de óbito ou cura |
| 4 | **Conclusão** — recomendações de saúde pública |

**Base:** SRAG 2023 / SIVEP-Gripe — Open Data SUS.
**Tamanho:** 279.453 notificações, 194 colunas, 22 MB em Parquet.
**Período-alvo:** todo o ano de 2023 (3.114 registros fora do escopo foram descartados).

---

# Pipeline em oito fases

```
Fase 0 — Setup e dicionário SIVEP-Gripe
Fase 1 — Profiling inicial       → notebook 01
Fase 2 — Tratamento e testes     → notebook 02 + cleaning.py + 29 testes pytest
Fase 3 — EDA descritiva          → notebook 03 + 19 figuras
Fase 4 — Feature engineering     → notebook 04 + features.py + 4 parquets
Fase 5 — Modelagem (Optuna)      → notebook 05 + models.py + best_model.joblib
Fase 6 — Avaliação e SHAP        → notebook 07 + 7 figuras
Fase 7 — Conclusões              → notebook 08 + PDFs + README
```

**Estrutura:** híbrida — notebooks para a narrativa e módulo `src/` para a lógica reusável e testável.

---

<!-- _class: section-divider -->

# 2. Decisões metodológicas

---

# Três decisões centrais

### 1. Split temporal (não aleatório)
- **Treino:** Jan-Jun 2023 (145 mil). **Teste:** Jul-Dez 2023 (103 mil).
- **Justificativa:** simula uso real do modelo e capta o shift VSR para COVID.

### 2. Vazamento controlado
- Excluídas: `DT_EVOLUCA`, `DT_ENCERRA` e `CRITERIO` (pós-desfecho).
- Mantidas: `UTI`, `SUPORT_VEN` e `RAIOX_RES` (clinicamente relevantes e pré-desfecho).

### 3. Métrica primária: AUC-PR (em vez de AUC-ROC)
- Óbito é a classe rara (cerca de 10%).
- A AUC-PR é mais sensível ao desempenho na classe minoritária.
- O baseline aleatório de AUC-PR equivale à taxa de óbito, em torno de 0,10.

---

<!-- _class: section-divider -->

# 3. Cinco insights da EDA

---

# Insight 1 — Pico de SRAG em 2023 foi pediátrico

![bg right:48% fit](figures/03_eda/02_series_semanais_agente.png)

- Pico em **maio/2023 (SE 21):** 35.248 casos.
- **Vetor principal:** VSR e Influenza em **menores de 5 anos** (não COVID).
- **Magnitude:** 42% dos casos SRAG 2023 estão em 0-4 anos.
- Mediana de idade: **8 anos**.
- COVID atinge pico na SE 43 (out/nov), com magnitude bem menor.

> O planejamento operacional pós-pandemia (UTI adulto, ventiladores adultos) está desalinhado em relação à carga real de 2023.

---

# Insight 2 — Gradiente etário fortíssimo

| Faixa | Letalidade |
|---|---|
| 0-4 | 1,2% |
| 5-19 | 1,5% |
| 20-39 | 8,9% |
| 40-59 | 17,0% |
| 60-69 | 21,4% |
| 70-79 | 23,9% |
| 80+ | 27,7% |

Variação de 23 vezes. Isoladamente, a idade é o preditor mais forte de mortalidade.

> SRAG é doença de duas pontas: incidência alta em crianças e letalidade alta em idosos.

---

# Insight 3 — Paradoxo de Simpson na vacinação COVID

![bg right:50% fit](figures/03_eda/13_paradoxo_simpson_resolvido.png)

**Marginal:** vacinados apresentam letalidade maior (22,5% contra 11,3%, diferença de 11pp). Trata-se de artefato.

**Estratificando por idade:** o efeito é protetor em todos os estratos.

| Faixa | Gap vacinado − não vacinado |
|---|---|
| 40-59 | −7,7pp |
| 60-69 | −5,2pp |
| 70-79 | −5,1pp |
| 80+ | −10,1pp |

> Justifica modelagem multivariada e cautela com tabelas marginais.

---

# Insight 4 — Suporte ventilatório invasivo é o melhor preditor

| `SUPORT_VEN` | Letalidade |
|---|---|
| Invasivo | 40,95% |
| Não invasivo | 7,44% |
| Não usou | 4,07% |

- UTI: 18,66% contra 5,79% (3,2 vezes a letalidade).
- Aparece como top-1 e top-2 no SHAP do modelo final.

> Observação: variável **hospitalar**. Reflete a resposta à gravidade observada, não apenas o estado de admissão.

---

# Insight 5 — Comorbidades raras e graves superam as comuns

Sem ajuste por idade: a cardiopatia parece muito letal (23,1% contra 13,9%, gap de 9pp).

Com ajuste por idade: em 70-79 anos, a cardiopatia apresenta 24,4% contra 26,8% sem cardiopatia. O gap inverte para −2,4pp.

| Comorbidade | Letalidade marginal | Prevalência |
|---|---|---|
| Hepática | 30,1% | rara |
| Renal | 26,6% | rara |
| Imunodepressão | 24,4% | rara |
| Diabetes | 23,6% | 16% |
| Cardiopatia | 23,1% | 49% |

> Modelos sem `IDADE_ANOS` inflam o peso de cardiopatia e diabetes. As comorbidades raras e graves apresentam efeito independente mais limpo.

---

<!-- _class: section-divider -->

# 4. Modelo final

---

# Oito modelos comparados — XGBoost com tuning como melhor escolha

| Modelo | AUC-PR (CV 5-fold) |
|---|---|
| Dummy (estratificado) | cerca de 0,10 |
| Regressão Logística | 0,49 |
| Random Forest tuned | 0,54 |
| **XGBoost tuned** | **0,55** |
| LightGBM tuned | 0,54 |

**Tuning:** Optuna TPE — Random Forest com 50 trials (60 mil); XGBoost e LightGBM com 80 trials (100 mil) cada; CV 3-fold por trial.

**Imbalance:** `class_weight='balanced'` (LR, RF, LGBM); `scale_pos_weight=9` (XGB).

---

# Métricas no teste temporal (Jul-Dez 2023)

![bg right:50% fit](figures/07_avaliacao/01_curvas_pr_roc.png)

| Métrica | Valor | Leitura |
|---|---|---|
| AUC-PR | 0,5531 | 5,4 vezes o baseline aleatório |
| AUC-ROC | 0,9055 | Discriminação elevada |
| Brier Score | 0,0915 | Calibração imperfeita |

**Casos no teste:** 103.072, com 10.517 óbitos reais.

**O threshold operacional depende do uso:**
- Youden (0,319): triagem (Sen=0,873).
- Recall ≥ 70% (0,515): alerta clínico (F1=0,526).

---

# Top features (SHAP)

![bg right:55% fit](figures/07_avaliacao/05_shap_beeswarm.png)

1. `SUPORT_VEN_ORD` — suporte invasivo.
2. `UTI` — internação em UTI.
3. `IDADE_ANOS` — gradiente monotônico.
4. Comorbidades raras e graves (hepática, renal e imunodepressão).
5. `SEM_NOT` — semana epidemiológica.

> O modelo reproduz a prática clínica: gravidade aguda (ventilação, UTI) supera vulnerabilidade biológica (idade, comorbidades), que por sua vez supera a pressão sazonal sobre o sistema.

---

<!-- _class: section-divider -->

# 5. Equidade e calibração

---

# Equidade — desempenho por subgrupo

![bg right:55% fit](figures/07_avaliacao/07_subgrupos.png)

- **Faixa etária:** a AUC-PR varia, mas o baseline acompanha (em todos os casos, acima do baseline).
- **Região:** desempenho uniforme nas cinco regiões.
- **Raça/cor:** sem disparidade dramática em categorias com n maior ou igual a 200.

> Disparidades remanescentes podem refletir disparidade no sinal disponível no registro (a qualidade do SIVEP varia por UF), e não necessariamente viés algorítmico direto.

---

# Calibração — quando não confiar nas probabilidades

![bg right:55% fit](figures/08_conclusoes/01_mapa_risco_idade_suporte.png)

O padrão visual é o mesmo nos dois painéis, mas as magnitudes divergem.

| Célula | Real | Predito |
|---|---|---|
| 80+ × Invasivo | 69,6% | 90,0% |
| 60-69 × Invasivo | 60,8% | 81,8% |
| 40-59 × Invasivo | 53,8% | 80,1% |

O modelo ranqueia muito bem, mas as probabilidades absolutas não são confiáveis. Recalibração (Platt scaling ou regressão isotônica) é recomendada antes de uso clínico.

---

<!-- _class: section-divider -->

# 6. Recomendações de saúde pública

---

# Cinco intervenções acionáveis

### 1. Reforço da capacidade pediátrica
Antes do pico anual de VSR (SE 17-21). Foco em UTI e leitos clínicos infantis.

### 2. Priorização vacinal por idade e comorbidade
Busca ativa de pacientes a partir de 70 anos com menos de quatro doses (gap real de 10pp em 80+).

### 3. Sistema de alerta antecipado semanal
Pipeline alimentando painel público (modelo InfoGripe). Antecipação de 2 a 4 semanas.

### 4. Padronização da qualidade do registro SIVEP-Gripe
Auditoria mensal por UF. Meta: menos de 30% de campos "Ignorado" em comorbidades-chave.

### 5. Pesquisa direcionada — `CLASSI_FIN=3` em maiores de 70 anos
Letalidade desproporcional (54 a 55%) sem agente identificado. Investigar coinfecções bacterianas.

---

<!-- _class: section-divider -->

# 7. Limitações e próximos passos

---

# Limitações conhecidas

1. **Qualidade do registro SIVEP-Gripe** — cerca de 70% de missing em comorbidades; os indicadores `_MISS` mitigam, mas não resolvem.

2. **Features hospitalares como proxy de gravidade** — `UTI` e `SUPORT_VEN` refletem a resposta ao quadro, não apenas o estado de admissão. Um modelo de triagem na chegada exige features mais restritas.

3. **Calibração imperfeita** — o Brier Score está próximo do baseline. As probabilidades absolutas não são confiáveis sem recalibração.

4. **Shift de distribuição** — o modelo foi treinado em 2023; um re-treinamento a cada seis meses é recomendado para uso contínuo.

5. **Equidade** — disparidades podem refletir disparidade no registro, e não no algoritmo. A preocupação não desaparece; muda a intervenção.

---

# Próximos passos

| Prioridade | Ação |
|---|---|
| Alta | Calibração (Platt ou isotônica) antes do uso clínico |
| Média | Modelo de triagem na chegada (sem features hospitalares) |
| Média | Pipeline em produção — Docker, API e dashboard de drift |
| Baixa | Validação prospectiva em 2024 (já disponível) |
| Baixa | Integração com InfoGripe — pipeline semanal público |
| Baixa | Fairness aprofundada — SHAP estratificado e métricas formais |

**Reprodutibilidade:** `uv sync && jupyter lab`; `random_state=42`; 29 testes pytest passando.

---

<!-- _class: title -->

# Síntese em uma frase

## 2023 foi um ano de SRAG **pediátrica em volume** e SRAG **idosa em letalidade**.

A maior alavanca de saúde pública não é melhorar mais 1 ou 2 pontos percentuais no modelo. É **(1)** vacinar idosos com cobertura de quatro ou mais doses, **(2)** reforçar a capacidade pediátrica antes do pico anual de VSR e **(3)** padronizar a qualidade do preenchimento do SIVEP-Gripe entre estados.

### Obrigado.

Repositório · Relatório técnico · Notebooks · `models/best_model.joblib`
*rodrigoeqv@gmail.com*
