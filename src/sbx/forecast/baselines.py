"""Простые эталонные модели и классические статистические модели."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import Context, Model, shift_period


class Naive(Model):
    name = "Naive"

    def predict(self, ctx, o, H):
        return np.repeat(ctx.Y[:, [o]], H, axis=1)


class SeasonalNaive(Model):
    """y(T+h) = y(T+h−12)."""
    name = "SeasonalNaive"

    def predict(self, ctx, o, H):
        idx = [o + h - 12 for h in range(1, H + 1)]
        out = np.full((ctx.S, H), np.nan)
        for j, k in enumerate(idx):
            if k >= 0:
                out[:, j] = ctx.Y[:, k]
        # если значения год назад нет — последнее наблюдение
        return np.where(np.isnan(out), ctx.Y[:, [o]], out)


class SeasonalNaiveGrowth(Model):
    """Сезонный наивный × темп роста г/г.

    Темп роста — медиана лог-приростов г/г ряда за последние 3 месяца; если истории МО
    ещё нет (точка прогноза в 2023 г.), используется прирост г/г общероссийского ряда
    СберИндекса той же категории. Если нет значения год назад — национальный сезонный
    профиль, применённый к последнему наблюдению.
    """
    name = "SNaive×Growth"

    def predict(self, ctx, o, H):
        Y = np.log(ctx.Y)
        yoy = [Y[:, o - k] - Y[:, o - k - 12] for k in range(3) if o - k - 12 >= 0]
        if yoy:
            g = np.nanmedian(np.vstack(yoy), axis=0)
        else:
            g = np.mean([ctx.nat_yoy(ctx.periods[o], lag=k) for k in range(3)], axis=0)
        origin = ctx.periods[o]
        fut = [shift_period(origin, h) for h in range(1, H + 1)]
        seas = ctx.nat_seasonal_matrix(origin, [origin] + fut)
        out = np.empty((ctx.S, H))
        for j in range(H):
            k = o + j + 1 - 12
            if k >= 0:
                out[:, j] = Y[:, k] + g
            else:
                out[:, j] = Y[:, o] + seas[:, j + 1] - seas[:, 0] + g * (j + 1) / 12
        return np.exp(out)


class NationalSeasonalDrift(Model):
    """Последнее значение × национальный сезонный профиль × тренд (прирост г/г / 12 в месяц)."""
    name = "NatSeasonal"

    def predict(self, ctx, o, H):
        origin = ctx.periods[o]
        fut = [shift_period(origin, h) for h in range(1, H + 1)]
        seas = ctx.nat_seasonal_matrix(origin, [origin] + fut)
        Y = np.log(ctx.Y)
        # уровень, очищенный от сезонности: среднее последних 3 месяцев
        hist = [shift_period(origin, -k) for k in range(3)]
        s_hist = ctx.nat_seasonal_matrix(origin, hist)
        level = np.nanmean(np.vstack([Y[:, o - k] - s_hist[:, k] for k in range(3) if o - k >= 0]), axis=0)
        g = ctx.nat_yoy(origin) / 12
        steps = np.arange(1, H + 1)
        return np.exp(level[:, None] + seas[:, 1:] + g[:, None] * (steps[None, :] + 1))


def _sf_forecast(Y_hist: np.ndarray, periods: list[str], H: int, which: str, sa: np.ndarray | None, fut_sa: np.ndarray | None):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    logY = np.log(Y_hist)
    if sa is not None:
        logY = logY - sa
    S, T = logY.shape
    ds = pd.PeriodIndex(periods, freq="M").to_timestamp()
    df = pd.DataFrame({"unique_id": np.repeat(np.arange(S), T), "ds": np.tile(ds, S), "y": logY.ravel()})
    model = {"ETS": AutoETS(season_length=12), "Theta": AutoTheta(season_length=12),
             "ARIMA": AutoARIMA(season_length=12)}[which]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = StatsForecast(models=[model], freq="MS", n_jobs=1).forecast(df=df, h=H)
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        out = fc[col].to_numpy().reshape(S, H)
    except NotImplementedError:   # слишком короткая история для модели — последнее значение (в лог-шкале, после очистки)
        out = np.repeat(logY[:, [-1]], H, axis=1)
    if fut_sa is not None:
        out = out + fut_sa
    return np.exp(out)


class StatsModel(Model):
    """AutoETS / AutoTheta / AutoARIMA (statsforecast) по логарифму ряда.

    seasonal_adjust=True: ряд предварительно очищается от сезонности национальными
    факторами СберИндекса (2019–origin), а сезонность возвращается в прогноз. Так короткий
    ряд МО (≤ 24 мес.) «наследует» многолетний сезонный профиль своей категории.
    """

    def __init__(self, which: str = "ETS", seasonal_adjust: bool = False, n_jobs: int = 8):
        self.which, self.sa, self.n_jobs = which, seasonal_adjust, n_jobs
        self.name = f"{which}{'-SA' if seasonal_adjust else ''}"

    def predict(self, ctx, o, H):
        hist = ctx.periods[: o + 1]
        Yh = ctx.Y[:, : o + 1]
        sa = fut = None
        if self.sa:
            origin = ctx.periods[o]
            sa = ctx.nat_seasonal_matrix(origin, hist)
            fut = ctx.nat_seasonal_matrix(origin, [shift_period(origin, h) for h in range(1, H + 1)])
        chunks = np.array_split(np.arange(ctx.S), self.n_jobs * 4)
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_sf_forecast)(Yh[c], hist, H, self.which, None if sa is None else sa[c], None if fut is None else fut[c])
            for c in chunks)
        return np.vstack(res)


def _prophet_chunk(Y_hist: np.ndarray, periods: list[str], H: int, params: dict) -> np.ndarray:
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet

    ds = pd.PeriodIndex(periods, freq="M").to_timestamp()
    fut = pd.DataFrame({"ds": pd.date_range(ds[-1], periods=H + 1, freq="MS")[1:]})
    out = np.empty((len(Y_hist), H))
    for i, y in enumerate(Y_hist):
        m = Prophet(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(pd.DataFrame({"ds": ds, "y": y}))
        out[i] = m.predict(fut).yhat.to_numpy()
    return out


class ProphetModel(Model):
    """Базовая модель организаторов: Prophet на уровнях ряда, параметры из конфигурации.

    При истории < 2 лет Prophet по умолчанию отключает годовую сезонность (yearly='auto'),
    поэтому фактически моделирует кусочно-линейный тренд — это и есть «как есть» baseline.
    """
    name = "Prophet"

    def __init__(self, params: dict | None = None, n_jobs: int = 8, name: str | None = None):
        p = dict(params or {})
        p.pop("enabled", None)
        self.params, self.n_jobs = p, n_jobs
        if name:
            self.name = name

    def predict(self, ctx, o, H):
        hist = ctx.periods[: o + 1]
        chunks = np.array_split(np.arange(ctx.S), self.n_jobs * 8)
        res = Parallel(n_jobs=self.n_jobs)(delayed(_prophet_chunk)(ctx.Y[c, : o + 1], hist, H, self.params) for c in chunks)
        return np.vstack(res)
