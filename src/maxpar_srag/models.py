"""Modelagem SRAG 2023 — Fase 5.

Construtores de pipeline, avaliação via CV e tuning Optuna.
Consumido pelo notebook 05_modelagem_baseline.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import optuna

from . import config
from .features import build_preprocessor

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODELS_DIR = config.PROJECT_ROOT / "models"
FIGURES_DIR_05 = config.FIGURES_DIR / "05_modelagem"

N_SPLITS_CV = 5
CV = StratifiedKFold(
    n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE
)

# XGBoost não suporta class_weight; scale_pos_weight = n_neg / n_pos.
# Taxa de óbito ~10% → scale_pos_weight ≈ 9.
XGB_SCALE_POS_WEIGHT = 9.0

SCORERS = {
    "ap": make_scorer(average_precision_score, response_method="predict_proba"),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
    "f1": make_scorer(f1_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}


# ---------------------------------------------------------------------------
# Avaliação por cross-validation
# ---------------------------------------------------------------------------

def evaluate_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: StratifiedKFold = CV,
    verbose: bool = True,
) -> dict[str, tuple[float, float]]:
    """CV 5-fold. Retorna {metric: (mean, std)}.

    n_jobs=1 no cross_validate evita conflito com paralelismo interno
    de LightGBM/XGBoost (nested parallelism instável no Windows).
    """
    results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, scoring=SCORERS,
        return_train_score=False,
        n_jobs=1,
    )
    out = {
        m: (results[f"test_{m}"].mean(), results[f"test_{m}"].std())
        for m in SCORERS
    }
    if verbose:
        print(f"  AUC-PR  : {out['ap'][0]:.4f} ± {out['ap'][1]:.4f}")
        print(f"  AUC-ROC : {out['roc_auc'][0]:.4f} ± {out['roc_auc'][1]:.4f}")
        print(f"  F1      : {out['f1'][0]:.4f} ± {out['f1'][1]:.4f}")
        print(f"  Recall  : {out['recall'][0]:.4f} ± {out['recall'][1]:.4f}")
    return out


# ---------------------------------------------------------------------------
# Construtores de pipeline
# ---------------------------------------------------------------------------

def build_dummy(feature_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", DummyClassifier(strategy="stratified", random_state=config.RANDOM_STATE)),
    ])


def build_logistic(feature_cols: list[str], C: float = 0.1) -> Pipeline:
    """LR com class_weight='balanced' e regularização L2.

    class_weight='balanced': pondera óbito (~10%) com peso ~9× — melhora
    recall na classe rara sem reamostragem.
    C=0.1: regularização moderada-forte para dados SIVEP com ruído de
    preenchimento e features correlacionadas (comorbidades × IDADE_ANOS).
    """
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=config.RANDOM_STATE,
        )),
    ])


def build_rf(feature_cols: list[str], **kwargs) -> Pipeline:
    """RF com class_weight='balanced_subsample'.

    balanced_subsample pondera classes dentro de cada bootstrap sample
    — mais robusto que 'balanced' global em datasets grandes (145k linhas).
    """
    params = dict(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", RandomForestClassifier(**params)),
    ])


def build_xgb(feature_cols: list[str], **kwargs) -> Pipeline:
    """XGBoost com scale_pos_weight para desequilíbrio de classes.

    scale_pos_weight = n_neg / n_pos ≈ 9: equivalente ao class_weight
    dos modelos sklearn. Multiplica o gradiente de amostras positivas.
    eval_metric='aucpr': alinha critério interno com métrica primária
    do projeto (usado em early stopping se ativado).
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
        eval_metric="aucpr",
        verbosity=0,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", xgb.XGBClassifier(**params)),
    ])


def build_lgbm(feature_cols: list[str], **kwargs) -> Pipeline:
    """LightGBM com class_weight='balanced' e bagging explícito.

    bagging_freq=1: necessário para ativar o subsampling de linhas
    quando subsample < 1.0 (sem isso, subsample é ignorado).
    """
    params = dict(
        n_estimators=300,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        bagging_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", lgb.LGBMClassifier(**params)),
    ])


# ---------------------------------------------------------------------------
# Auxiliares do Optuna
# ---------------------------------------------------------------------------

def _cv3_ap(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """AUC-PR médio em 3-fold CV — mais rápido para triagem de hiperparâmetros.

    3 folds preservam a ordenação relativa dos hiperparâmetros com custo
    menor que 5 folds. O modelo final é avaliado com 5 folds (evaluate_cv).
    """
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    scores = []
    for tr, va in cv3.split(X, y):
        p = clone(pipeline)
        p.fit(X.iloc[tr], y.iloc[tr])
        proba = p.predict_proba(X.iloc[va])[:, 1]
        scores.append(average_precision_score(y.iloc[va], proba))
    return float(np.mean(scores))


def _subsample(
    X: pd.DataFrame, y: pd.Series, n: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Subsample posicional para acelerar Optuna."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def _make_study() -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )


def _log_progress(n_trials: int):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        t = trial.number + 1
        if t == 1 or t % 10 == 0 or t == n_trials:
            print(f"  Trial {t:3d}/{n_trials} | melhor AUC-PR = {study.best_value:.4f}")
    return callback


# ---------------------------------------------------------------------------
# Tuning Optuna por modelo
# ---------------------------------------------------------------------------

def tune_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    *,
    n_trials: int = 50,
    n_sample: int = 60_000,
) -> optuna.Study:
    """Optuna para Random Forest (TPE, subsample=30k, CV 3-fold)."""
    Xs, ys = _subsample(X_train, y_train, n_sample)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features": trial.suggest_float("max_features", 0.3, 0.9),
        }
        return _cv3_ap(build_rf(feature_cols, **params), Xs, ys)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_progress(n_trials)])
    return study


def tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    *,
    n_trials: int = 80,
    n_sample: int = 100_000,
) -> optuna.Study:
    """Optuna para XGBoost (TPE, subsample=60k, CV 3-fold)."""
    Xs, ys = _subsample(X_train, y_train, n_sample)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return _cv3_ap(build_xgb(feature_cols, **params), Xs, ys)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_progress(n_trials)])
    return study


def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
    *,
    n_trials: int = 80,
    n_sample: int = 100_000,
) -> optuna.Study:
    """Optuna para LightGBM (TPE, subsample=60k, CV 3-fold)."""
    Xs, ys = _subsample(X_train, y_train, n_sample)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "bagging_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return _cv3_ap(build_lgbm(feature_cols, **params), Xs, ys)

    study = _make_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[_log_progress(n_trials)])
    return study


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_pr_curves(
    fitted_pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    save_path: Optional[Path] = None,
):
    """Curvas Precision-Recall no conjunto de teste."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_ap = float(y_test.mean())
    ax.axhline(
        y=baseline_ap, color="gray", linestyle="--", linewidth=1.2,
        label=f"Aleatório (AP={baseline_ap:.3f})",
    )
    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, linewidth=1.8, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall (Sensibilidade — classe Óbito)", fontsize=11)
    ax.set_ylabel("Precisão", fontsize=11)
    ax.set_title("Curvas Precision-Recall — Teste Jul–Dez 2023", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    fitted_pipelines: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    save_path: Optional[Path] = None,
):
    """Curvas ROC no conjunto de teste."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Aleatório")
    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("Taxa de Falso Positivo (1 − Especificidade)", fontsize=11)
    ax.set_ylabel("Taxa de Verdadeiro Positivo (Recall)", fontsize=11)
    ax.set_title("Curvas ROC — Teste Jul–Dez 2023", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, name: str) -> Path:
    """Serializa pipeline treinado em models/<name>.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    print(f"Modelo salvo: {path}")
    return path


def load_model(name: str) -> Pipeline:
    """Carrega pipeline de models/<name>.joblib."""
    return joblib.load(MODELS_DIR / f"{name}.joblib")
