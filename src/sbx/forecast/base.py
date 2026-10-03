"""Контекст бэктеста и общий интерфейс моделей.

Защита от утечки: модель получает матрицу Y, в которой все значения после точки прогноза
заменены на NaN, а внешние ряды доступны только через `asof` (данные не позже origin).
Известные заранее ковариаты (календарь, климатическая норма) берутся без ограничения.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import CAT_CODE, NATIONAL_TYPE, national_seasonal_factors


def shift_period(p: str, k: int) -> str:
    return str(pd.Period(p, "M") + k)


@dataclass
class Context:
    Y: np.ndarray                    # S × T, уровни расходов (руб.)
    periods: list[str]               # T периодов 'YYYY-MM'
    meta: pd.DataFrame               # S строк: territory_id, category, region_key, статические признаки
    national: pd.DataFrame           # общероссийские ряды по периодам
    region: pd.DataFrame             # ИПЦ/зарплаты по субъектам
    neighbors: pd.DataFrame          # соседи по дорогам
    weather: pd.DataFrame | None = None
    news: pd.DataFrame | None = None  # признаки новостей region_key × period (+ territory_id)
    cfg: dict = field(default_factory=dict)
    cpd_asof: pd.DataFrame | None = None  # признаки детектора изменений, вычисленные «на момент t»
    _seas_cache: dict = field(default_factory=dict)

    @property
    def S(self):
        return self.Y.shape[0]

    def masked(self, o: int) -> "Context":
        """Копия контекста, где будущие (после индекса o) значения скрыты."""
        Ym = self.Y.copy()
        Ym[:, o + 1:] = np.nan
        return Context(Ym, self.periods, self.meta, self.national, self.region, self.neighbors,
                       self.weather, self.news, self.cfg, self.cpd_asof, self._seas_cache)

    def asof(self, df: pd.DataFrame, origin: str, col: str = "period") -> pd.DataFrame:
        return df[df[col] <= origin]

    def nat_seasonal(self, origin: str) -> pd.DataFrame:
        """Сезонные факторы РФ (лог) по данным не позже origin: месяц × категория конкурса."""
        if origin not in self._seas_cache:
            f = national_seasonal_factors(self.national, origin).pivot(index="month", columns="type", values="seas")
            self._seas_cache[origin] = pd.DataFrame({cat: f[t] for cat, t in NATIONAL_TYPE.items()})
        return self._seas_cache[origin]

    def nat_seasonal_matrix(self, origin: str, periods: list[str]) -> np.ndarray:
        """S × len(periods) матрица национальных сезонных факторов для каждого ряда."""
        f = self.nat_seasonal(origin)
        months = [int(p[5:7]) for p in periods]
        by_cat = {cat: f.loc[months, cat].to_numpy() for cat in f.columns}
        return np.vstack([by_cat[c] for c in self.meta.category])

    def nat_yoy(self, origin: str, lag: int = 0) -> np.ndarray:
        """Лог-прирост г/г общероссийского ряда (соответствующий тип) на месяц origin−lag."""
        nat = self.national.set_index("period")
        p, p12 = shift_period(origin, -lag), shift_period(origin, -lag - 12)
        vals = {cat: np.log(nat.loc[p, f"nat_spend::{t}"] / nat.loc[p12, f"nat_spend::{t}"]) for cat, t in NATIONAL_TYPE.items()}
        return self.meta.category.map(vals).to_numpy(float)


class Model:
    name = "model"
    quantiles: tuple = ()

    def predict(self, ctx: Context, o: int, H: int) -> np.ndarray:
        """Возвращает массив S × H (шаги 1..H) или dict {'mean': S×H, 'q0.1': …}."""
        raise NotImplementedError


def build_context(processed, external, cfg, min_obs: int = 24) -> Context:
    from ..data import weather_monthly, news_region_monthly
    panel = pd.read_parquet(f"{processed}/panel.parquet")
    static = pd.read_parquet(f"{processed}/static.parquet")
    wide = panel.pivot_table(index=["territory_id", "category"], columns="period", values="value")
    wide = wide[wide.notna().sum(axis=1) >= min_obs]
    periods = list(wide.columns)
    meta = wide.index.to_frame(index=False).merge(static, on="territory_id", how="left")
    meta["cat_code"] = meta.category.map(CAT_CODE)
    return Context(
        Y=wide.to_numpy(float), periods=periods, meta=meta,
        national=pd.read_parquet(f"{processed}/exog_national.parquet"),
        region=pd.read_parquet(f"{processed}/exog_region.parquet"),
        neighbors=pd.read_parquet(f"{processed}/neighbors.parquet"),
        weather=weather_monthly(external), news=news_region_monthly(processed), cfg=cfg,
        cpd_asof=pd.read_parquet(f"{processed}/cpd_asof.parquet") if Path(f"{processed}/cpd_asof.parquet").exists() else None)
