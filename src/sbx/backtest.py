"""Бэктест со скользящей точкой прогноза (rolling origin, expanding window).

Для каждой точки прогноза T ∈ [first_origin, last_origin] модель видит только данные ≤ T
и прогнозирует шаги h = 1..min(12, конец данных − T).

Оценка:
* по шагу h — все прогнозы ровно на h месяцев вперёд;
* по окну H ∈ {1, 3, 6, 12} — средняя ошибка на шагах 1..H по всем точкам T, для которых
  окно целиком попадает в данные (H=12 — прогноз всего 2024 г. из декабря 2023 г.).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .forecast.base import Context, Model


def origins(ctx: Context, first: str, last: str) -> list[int]:
    return [i for i, p in enumerate(ctx.periods) if first <= p <= last]


def run_model(model: Model, ctx: Context, origin_idx: list[int], max_h: int = 12, verbose: bool = True) -> pd.DataFrame:
    T = len(ctx.periods)
    frames, importances = [], []
    for o in origin_idx:
        H = min(max_h, T - 1 - o)
        if H < 1:
            continue
        t0 = time.time()
        res = model.predict(ctx.masked(o), o, H)
        out = res if isinstance(res, dict) else {"mean": res}
        S = ctx.S
        df = pd.DataFrame({"s": np.repeat(np.arange(S), H), "origin": o, "h": np.tile(np.arange(1, H + 1), S)})
        for k, v in out.items():
            df["yhat" if k == "mean" else k] = np.asarray(v, float).reshape(-1)
        frames.append(df)
        imp = getattr(model, "last_importance", None)
        if imp is not None:
            importances.append(imp.rename(ctx.periods[o]))
        if verbose:
            print(f"  {model.name:<22} origin={ctx.periods[o]}  H={H:<2}  {time.time() - t0:6.1f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["importance"] = pd.concat(importances, axis=1) if importances else None
    return out


def attach_truth(fc: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    fc = fc.copy()
    t = fc.origin.to_numpy() + fc.h.to_numpy()
    s = fc.s.to_numpy()
    fc["y"] = ctx.Y[s, t]
    fc["y_origin"] = ctx.Y[s, fc.origin.to_numpy()]
    # масштаб MASE: средняя |разность первого порядка| ряда на истории до точки прогноза
    d = np.abs(np.diff(ctx.Y, axis=1))
    csum = np.concatenate([np.zeros((ctx.S, 1)), np.nancumsum(d, axis=1)], axis=1)
    o = fc.origin.to_numpy()
    fc["scale"] = csum[s, o] / np.maximum(o, 1)
    fc["target_period"] = np.array(ctx.periods)[t]
    fc["origin_period"] = np.array(ctx.periods)[o]
    fc["category"] = ctx.meta.category.to_numpy()[s]
    return fc


def window_view(fc: pd.DataFrame, windows=(1, 3, 6, 12), T: int = 24) -> pd.DataFrame:
    """Строки прогнозов, участвующие в оценке окна H (окно должно целиком лежать в данных)."""
    parts = []
    for H in windows:
        sub = fc[(fc.h <= H) & (fc.origin + H <= T - 1)]
        parts.append(sub.assign(window=H))
    return pd.concat(parts, ignore_index=True)


def evaluate(fc: pd.DataFrame, T: int = 24, windows=(1, 3, 6, 12)) -> dict[str, pd.DataFrame]:
    w = window_view(fc, windows, T)
    return {
        "by_window": metrics.summarize(w, ["model", "window"]),
        "by_window_category": metrics.summarize(w, ["model", "window", "category"]),
        "by_step": metrics.summarize(fc, ["model", "h"]),
        "by_origin": metrics.summarize(fc[fc.h == 1], ["model", "origin_period"]),
    }


def save_forecasts(fc: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path)
