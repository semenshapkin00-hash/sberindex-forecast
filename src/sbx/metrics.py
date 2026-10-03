"""Метрики качества прогноза. MAE — основная (обязательная по условиям конкурса)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def r2(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2))


def wape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sum(np.abs(y - f)) / np.sum(np.abs(y)))


def mape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(np.abs(y - f) / np.abs(y)))


def smape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f))))


def summarize(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """df: столбцы y, yhat, scale (масштаб MASE) [+ by]."""
    def agg(g):
        return pd.Series({
            "MAE": mae(g.y, g.yhat), "RMSE": rmse(g.y, g.yhat), "R2": r2(g.y, g.yhat),
            "WAPE": wape(g.y, g.yhat), "MAPE": mape(g.y, g.yhat), "sMAPE": smape(g.y, g.yhat),
            "MASE": float(np.mean(np.abs(g.y - g.yhat) / g.scale)) if "scale" in g else np.nan,
            # R² по логарифмическим приростам от точки прогноза: насколько модель объясняет
            # именно динамику, а не межмуниципальные различия уровней
            "R2_dyn": r2(np.log(g.y / g.y_origin), np.log(np.maximum(g.yhat, 1.0) / g.y_origin)) if "y_origin" in g else np.nan,
            "n": len(g),
        })
    if not by:
        return agg(df).to_frame().T
    return df.groupby(by).apply(agg, include_groups=False).reset_index()


def paired_test(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Сравнение двух моделей по абсолютным ошибкам, усреднённым по ряду (пары рядов).

    Возвращает разность MAE (a − b), 95% бутстреп-интервал и p-value критерия Уилкоксона.
    Усреднение внутри ряда устраняет зависимость ошибок по горизонтам/точкам прогноза
    (вариант теста Диболда–Мариано для панели).
    """
    a, b = np.asarray(err_a), np.asarray(err_b)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    w = stats.wilcoxon(a, b) if np.any(d != 0) else None
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)), "p_wilcoxon": float(w.pvalue) if w else 1.0,
            "share_better": float(np.mean(d < 0))}
