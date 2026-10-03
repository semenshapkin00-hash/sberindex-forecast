"""Глобальная модель LightGBM с прямым многошаговым прогнозом.

Одна модель на все ряды и горизонты: наблюдение = (ряд s, момент t, шаг h), цель —
лог-прирост log y[t+h] − log y[t]. Обучение в точке прогноза T использует только пары с
t + h ≤ T. Вес наблюдения = y[t]: тогда L1 по лог-приросту ≈ MAE в рублях, т.е. модель
оптимизирует ту же метрику, по которой оценивается.

Группы признаков (feature_sets):
  base      — лаги и сезонные «якоря» самого ряда, волатильность, категория, шаг, месяц
  regional  — национальный сезонный профиль СберИндекса (2019+), приросты по РФ и по панели,
              ИПЦ/зарплаты/задолженность субъекта, ключевая ставка, статические признаки МО
  calendar  — рабочие/праздничные дни целевого месяца (известны заранее)
  weather   — аномалии температуры и осадков в точке прогноза
  spatial   — динамика соседних МО по дорожной сети (connection.parquet)
  news      — индексы новостей по субъекту/МО на момент t (только опубликованные до конца t)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Context, Model, shift_period

STATIC = ["log_population", "urban_share", "market_access", "is_capital", "mo_type", "monotown_status", "lat", "lon"]


def _period_index(periods: list[str], extra: int = 12) -> list[str]:
    return periods + [shift_period(periods[-1], k) for k in range(1, extra + 1)]


class FeatureBuilder:
    def __init__(self, ctx: Context, o: int, feature_sets: list[str], use_news: bool, use_region: bool = True):
        self.ctx, self.o, self.sets, self.use_news = ctx, o, set(feature_sets), use_news and ctx.news is not None
        self.use_region = use_region
        self.origin = ctx.periods[o]
        self.L = np.log(ctx.Y)                       # S × T, будущее уже NaN (ctx.masked)
        self.S, self.T = self.L.shape
        self.all_periods = _period_index(ctx.periods)
        self.cat = ctx.meta.category.astype("category").cat.codes.to_numpy()
        self._prep()

    # --------------------------------------------------------------- подготовка
    def _prep(self):
        ctx, L = self.ctx, self.L
        # панельные (кросс-секционные) медианы по категории на каждый месяц — только из истории
        self.cs_mom = np.full((self.S, self.T), np.nan)
        self.cs_yoy = np.full((self.S, self.T), np.nan)
        d1 = L - np.roll(L, 1, axis=1)
        d1[:, 0] = np.nan
        d12 = np.full_like(L, np.nan)
        d12[:, 12:] = L[:, 12:] - L[:, :-12]
        self.d1, self.d12 = d1, d12
        for c in np.unique(self.cat):
            m = self.cat == c
            with np.errstate(all="ignore"):
                self.cs_mom[m] = np.nanmedian(d1[m], axis=0)
                self.cs_yoy[m] = np.nanmedian(d12[m], axis=0)
        # региональная медиана г/г по той же категории
        reg = ctx.meta.region_key.astype("category").cat.codes.to_numpy()
        key = reg * 10 + self.cat
        self.reg_yoy = np.full_like(L, np.nan)
        self.reg_mom3 = np.full_like(L, np.nan)
        d3 = np.full_like(L, np.nan)
        d3[:, 3:] = L[:, 3:] - L[:, :-3]
        df12 = pd.DataFrame(d12).groupby(key).transform("median").to_numpy()
        df3 = pd.DataFrame(d3).groupby(key).transform("median").to_numpy()
        self.reg_yoy, self.reg_mom3, self.d3 = df12, df3, d3
        # соседи (взвешенное среднее по дорожной близости)
        if "spatial" in self.sets:
            nb = ctx.neighbors
            meta = ctx.meta[["territory_id", "category"]].reset_index().rename(columns={"index": "row"})
            j = meta.merge(nb, on="territory_id").merge(meta.rename(columns={"territory_id": "neighbor_id", "row": "nrow"}),
                                                        on=["neighbor_id", "category"])
            self.nb_pairs = j[["row", "nrow", "weight"]].to_numpy()
            self.nb_d3 = self._nb_avg(d3)
            self.nb_d12 = self._nb_avg(d12)
        # национальный сезонный профиль на период (по данным ≤ origin)
        self.seas = ctx.nat_seasonal_matrix(self.origin, self.all_periods)  # S × (T+12)
        nat = ctx.national.set_index("period")
        self.nat_yoy = np.zeros((self.S, self.T))
        for k, p in enumerate(ctx.periods):
            if p <= self.origin:
                self.nat_yoy[:, k] = ctx.nat_yoy(p)
            else:
                self.nat_yoy[:, k] = np.nan
        cal = nat.reindex(self.all_periods)
        self.cal = {c: cal[c].to_numpy(float) for c in ["n_workdays", "n_holidays", "n_days", "n_fridays"]}
        self.key_rate = nat.reindex(self.all_periods)["key_rate_mean"].to_numpy(float)
        # региональная макроэкономика
        r = ctx.region.copy()
        r = r[r.period <= self.origin]
        r = r.sort_values("period")
        r["cpi_log"] = np.log(r.cpi_total / 100)
        r["cpi_yoy"] = r.groupby("region_key").cpi_log.transform(lambda s: s.rolling(12, min_periods=12).sum())
        r["cpi_3m"] = r.groupby("region_key").cpi_log.transform(lambda s: s.rolling(3).sum())
        r["arrears_log"] = np.log1p(r.wage_arrears)
        piv = {c: r.pivot_table(index="region_key", columns="period", values=c).reindex(columns=self.ctx.periods)
               for c in ["cpi_yoy", "cpi_3m", "wage_yoy", "arrears_log"]}
        rk = ctx.meta.region_key.to_numpy()
        self.reg_macro = {c: v.reindex(rk).to_numpy(float) for c, v in piv.items()}
        # погода
        if "weather" in self.sets and ctx.weather is not None:
            w = ctx.weather[ctx.weather.period <= self.origin]
            tid = ctx.meta.territory_id.to_numpy()
            self.weather = {c: w.pivot_table(index="territory_id", columns="period", values=c)
                            .reindex(index=tid, columns=ctx.periods).to_numpy(float)
                            for c in ["t2m_anom", "prectotcorr_anom", "snodp"] if c in w}
        else:
            self.weather = {}
        # детектор изменений «на момент t» (связка CPD → прогноз)
        self.cpdf = {}
        if "cpd" in self.sets and ctx.cpd_asof is not None:
            c = ctx.cpd_asof
            c = c[c.period <= self.origin]
            idx = pd.MultiIndex.from_frame(ctx.meta[["territory_id", "category"]])
            for col in ["cp_age", "cp_mag", "cp_prob"]:
                p = c.pivot_table(index=["territory_id", "category"], columns="period", values=col)
                self.cpdf[col] = p.reindex(index=idx, columns=ctx.periods).to_numpy(float)
        # новости (только опубликованные не позже конца месяца точки прогноза)
        self.news = {}
        if self.use_news:
            nw = ctx.news[ctx.news.period <= self.origin]
            reg_rows = nw[nw.territory_id.isna() & (nw.region_key != "__RU__")]
            reg_cols = [c for c in reg_rows.columns if c.startswith("news_") and reg_rows[c].notna().any()
                        and not c.startswith("news_nat_") and not c.endswith("_mo")]
            for c in reg_cols:
                p = reg_rows.pivot_table(index="region_key", columns="period", values=c).reindex(columns=ctx.periods)
                self.news[c] = p.reindex(rk).to_numpy(float)
            mo_rows = nw[nw.territory_id.notna()]
            tid = ctx.meta.territory_id.to_numpy()
            for c in ["news_n_mo", "news_shock_mo"]:
                if c in mo_rows:
                    p = mo_rows.pivot_table(index="territory_id", columns="period", values=c).reindex(index=tid, columns=ctx.periods)
                    self.news[c] = np.nan_to_num(p.to_numpy(float))
            nat = nw[nw.region_key == "__RU__"].set_index("period").reindex(ctx.periods)
            topic = {"Маркетплейсы": "market", "Транспорт": "transport", "Здоровье": "health",
                     "Общественное питание": "horeca", "Продовольствие": "food"}
            cats = ctx.meta.category.to_numpy()
            M = np.zeros((self.S, self.T))
            for cat, code in topic.items():
                col = f"news_nat_topic_{code}"
                if col in nat:
                    M[cats == cat] = nat[col].to_numpy(float)
            self.news["news_nat_cat"] = M
            if "news_nat_shock_score" in nat:
                self.news["news_nat_shock"] = np.tile(nat["news_nat_shock_score"].to_numpy(float), (self.S, 1))

    def _nb_avg(self, M: np.ndarray) -> np.ndarray:
        rows, nrows, w = self.nb_pairs[:, 0].astype(int), self.nb_pairs[:, 1].astype(int), self.nb_pairs[:, 2]
        num = np.zeros_like(M)
        den = np.zeros_like(M)
        vals = M[nrows]
        ok = ~np.isnan(vals)
        np.add.at(num, rows, np.where(ok, vals, 0) * w[:, None])
        np.add.at(den, rows, ok * w[:, None])
        with np.errstate(all="ignore"):
            return num / den

    # --------------------------------------------------------------- признаки
    def features(self, s: np.ndarray, t: np.ndarray, h: np.ndarray) -> pd.DataFrame:
        L, f = self.L, {}
        lv = L[s, t]
        g = lambda M, tt: np.where((tt >= 0) & (tt < M.shape[1]), M[s, np.clip(tt, 0, M.shape[1] - 1)], np.nan)
        f["h"] = h
        f["cat"] = self.cat[s]
        f["target_month"] = np.array([int(p[5:7]) for p in self.all_periods])[t + h]
        f["origin_month"] = np.array([int(p[5:7]) for p in self.all_periods])[t]
        f["level"] = lv
        # лаги очищены от национальной сезонности: тогда закономерности, выученные на
        # точках отсчёта в одни месяцы, переносятся на другие (в т.ч. на декабрь)
        sz = self.seas
        sa = lambda tt: np.where(tt >= 0, sz[s, np.clip(tt, 0, None)], np.nan) - sz[s, t]
        for k in range(1, 12):
            f[f"lag{k}"] = g(L, t - k) - lv - sa(t - k)
        for k in (1, 2, 3, 12):
            f[f"lag{k}_raw"] = g(L, t - k) - lv
        f["ma3"] = np.nanmean(np.vstack([g(L, t - k) - sa(t - k) for k in range(3)]), axis=0) - lv
        f["anchor"] = g(L, t + h - 12) - lv - sa(t + h - 12)      # год назад в целевом месяце (SA)
        f["anchor_raw"] = g(L, t + h - 12) - lv
        f["anchor_prev"] = g(L, t + h - 12) - g(L, t - 12)        # прошлогодний путь от месяца t до целевого
        f["anchor_prev_vs_nat"] = f["anchor_prev"] - (sz[s, t + h] - sz[s, t])
        f["yoy0"] = g(self.d12, t)
        f["yoy_avg3"] = np.nanmean(np.vstack([g(self.d12, t - k) for k in range(3)]), axis=0)
        f["mom1"] = g(self.d1, t) - (sz[s, t] - np.where(t >= 1, sz[s, np.maximum(t - 1, 0)], np.nan))
        f["mom3"] = g(self.d3, t) - (sz[s, t] - np.where(t >= 3, sz[s, np.maximum(t - 3, 0)], np.nan))
        f["vol"] = self._vol_fast(s, t)
        if "regional" in self.sets:
            f["nat_seas_diff"] = self.seas[s, t + h] - self.seas[s, t]
            f["nat_seas_target"] = self.seas[s, t + h]
            f["nat_yoy"] = g(self.nat_yoy, t)
            f["cs_mom"] = g(self.cs_mom, t)
            f["cs_yoy"] = g(self.cs_yoy, t)
            f["reg_yoy"] = g(self.reg_yoy, t)
            f["reg_mom3"] = g(self.reg_mom3, t)
            f["own_vs_reg_yoy"] = f["yoy0"] - f["reg_yoy"]
            for c, M in self.reg_macro.items():
                f[c] = g(M, t)
            f["key_rate"] = self.key_rate[t]
            f["key_rate_d3"] = self.key_rate[t] - self.key_rate[np.maximum(t - 3, 0)]
            meta = self.ctx.meta
            for c in STATIC:
                f[c] = meta[c].to_numpy(float)[s]
            if self.use_region:
                f["region"] = meta.region_code.to_numpy()[s]
        if "calendar" in self.sets:
            for c, v in self.cal.items():
                f[f"{c}_target"] = v[t + h]
            f["workdays_diff"] = self.cal["n_workdays"][t + h] - self.cal["n_workdays"][t]
        if "spatial" in self.sets:
            f["nb_mom3"] = g(self.nb_d3, t)
            f["nb_yoy"] = g(self.nb_d12, t)
            f["own_vs_nb_mom3"] = f["mom3"] - f["nb_mom3"]
        for c, M in self.weather.items():
            f[f"w_{c}"] = g(M, t)
        for c, M in self.cpdf.items():
            f[c] = g(M, t)
        for c, M in self.news.items():
            f[c] = g(M, t)
            f[c + "_l1"] = g(M, t - 1)
        return pd.DataFrame(f)

    def _vol_fast(self, s, t):
        out = np.full(len(s), np.nan)
        for tt in np.unique(t):
            m = t == tt
            if tt >= 2:
                out[m] = np.nanstd(self.d1[s[m], 1: tt + 1], axis=1)
        return out


class GlobalGBM(Model):
    name = "LightGBM"

    def __init__(self, cfg: dict, use_news: bool = True, feature_sets: list[str] | None = None, name: str | None = None,
                 max_train_rows: int = 1_500_000, seed: int = 42, fm_cache: str | None = None):
        self.cfg = cfg
        self.fm_cache = fm_cache   # путь к прогнозам Chronos-2 (признаки стэкинга) или None
        self.use_news = use_news
        self.sets = feature_sets or cfg.get("feature_sets", ["base", "regional", "calendar", "weather", "spatial", "cpd", "news"])
        if not use_news:
            self.sets = [x for x in self.sets if x != "news"]
        self.max_rows, self.seed = max_train_rows, seed
        if name:
            self.name = name
        self.last_importance: pd.Series | None = None

    def _add_fm(self, X: pd.DataFrame, ctx: Context, s, t, h) -> None:
        if not hasattr(self, "_fm"):
            fm = pd.read_parquet(self.fm_cache)
            key = ctx.meta[["territory_id", "category"]].reset_index().rename(columns={"index": "s"})
            fm = fm.merge(key, on=["territory_id", "category"])
            fm["t"] = fm.origin_period.map({p: i for i, p in enumerate(ctx.periods)})
            self._fm = fm.set_index(["s", "t", "h"])[["fm", "fm_width"]]
        idx = pd.MultiIndex.from_arrays([s, t, h])
        vals = self._fm.reindex(idx)
        X["fm"] = vals.fm.to_numpy()
        X["fm_width"] = vals.fm_width.to_numpy()

    def _offset(self, fb: "FeatureBuilder", s, t, h) -> np.ndarray:
        """Структурная база: ожидаемое лог-изменение по национальному сезонному профилю
        СберИндекса (2019+) и тренду г/г. Деревья учат только поправку к ней — это важно для
        дальних горизонтов, где у панели ещё нет годового «якоря»."""
        if self.cfg.get("offset", "national") != "national":
            return np.zeros(len(s))
        g = fb.nat_yoy[s, np.minimum(t, fb.o)]
        return fb.seas[s, t + h] - fb.seas[s, t] + np.nan_to_num(g) * h / 12

    def predict(self, ctx: Context, o: int, H: int):
        import lightgbm as lgb

        fb = FeatureBuilder(ctx, o, self.sets, self.use_news, self.cfg.get("region_feature", True))
        S = ctx.S
        # обучающие тройки (s, t, h) с t + h ≤ o
        tr = [(t, h) for t in range(o) for h in range(1, 13) if t + h <= o]
        tt = np.array([x[0] for x in tr])
        hh = np.array([x[1] for x in tr])
        s_tr = np.repeat(np.arange(S), len(tr))
        t_tr = np.tile(tt, S)
        h_tr = np.tile(hh, S)
        rng = np.random.default_rng(self.seed + o)
        if len(s_tr) > self.max_rows:
            keep = rng.choice(len(s_tr), self.max_rows, replace=False)
            s_tr, t_tr, h_tr = s_tr[keep], t_tr[keep], h_tr[keep]
        X = fb.features(s_tr, t_tr, h_tr)
        if self.fm_cache:
            self._add_fm(X, ctx, s_tr, t_tr, h_tr)
        y = fb.L[s_tr, t_tr + h_tr] - fb.L[s_tr, t_tr]
        w = ctx.Y[s_tr, t_tr]
        off = self._offset(fb, s_tr, t_tr, h_tr)
        ok = np.isfinite(y)
        params = dict(self.cfg["params"])
        params["seed"] = self.seed
        cat_feats = [c for c in ("cat", "region") if c in X]
        ds = lgb.Dataset(X[ok], y[ok], weight=w[ok] / np.nanmean(w[ok]), init_score=off[ok],
                         categorical_feature=cat_feats, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=self.cfg["num_boost_round"])
        self.last_importance = pd.Series(booster.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        # прогноз из точки o
        s_te = np.repeat(np.arange(S), H)
        h_te = np.tile(np.arange(1, H + 1), S)
        t_te = np.full_like(s_te, o)
        Xte = fb.features(s_te, t_te, h_te)
        if self.fm_cache:
            self._add_fm(Xte, ctx, s_te, t_te, h_te)
        pred = booster.predict(Xte) + self._offset(fb, s_te, t_te, h_te)
        yhat = np.exp(fb.L[s_te, o] + pred)
        return yhat.reshape(S, H)
