"""Онлайн-ансамбли прогнозов.

Веса участников пересчитываются в каждой точке прогноза T по ошибкам их прошлых прогнозов,
чьи факты известны к T (без утечки): w ∝ 1/MAE по той же категории и шагу h; если для шага h
истории нет (например, h = 12 в декабре 2023 г.), берётся ближайший шаг, для которого она есть.
Историю на первой тестовой точке дают прогнозы из валидационных точек июня–ноября 2023 г.

Итоговая схема («Ансамбль») объединяет модели, обучаемые на панели (варианты LightGBM),
и сезонные модели. Для анализа чувствительности считаются и другие схемы: выбор участников
по валидации, онлайн-выбор лучших, все кандидаты, медиана.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _Panel:
    def __init__(self, fc: pd.DataFrame, models: list[str]):
        sub = fc[fc.model.isin(models)]
        wide = sub.pivot_table(index=["s", "origin", "h"], columns="model", values="yhat")[models].dropna()
        self.keys = wide.index.to_frame(index=False)
        self.W = wide.to_numpy()
        truth = sub.drop_duplicates(["s", "origin", "h"]).set_index(["s", "origin", "h"]).y.reindex(wide.index).to_numpy()
        self.ae = np.abs(self.W - truth[:, None])
        self.cat = sub.drop_duplicates("s").set_index("s").category.reindex(self.keys.s).to_numpy()
        self.origin = self.keys.origin.to_numpy()
        self.h = self.keys.h.to_numpy()
        self.target = self.origin + self.h
        self.models = models

    def combine(self, first: int, choose=None, power: float = 1.0, min_obs: int = 200):
        """choose(mae) -> индексы участников (по умолчанию все); веса ∝ 1/MAE^power."""
        out = np.full(len(self.W), np.nan)
        log = []
        for o in sorted(set(self.origin[self.origin >= first])):
            cur, past = self.origin == o, self.target <= o
            for c in np.unique(self.cat[cur]):
                in_c = self.cat == c
                hist_h = np.unique(self.h[past & in_c])
                for h in np.unique(self.h[cur & in_c]):
                    m_cur = cur & in_c & (self.h == h)
                    if len(hist_h):
                        hn = hist_h[np.argmin(np.abs(hist_h - h))]     # ближайший шаг с известной историей
                        m_past = past & in_c & (self.h == hn)
                    else:
                        m_past = past & in_c
                    mae = self.ae[m_past].mean(axis=0) if m_past.sum() >= min_obs else np.ones(len(self.models))
                    sel = np.arange(len(self.models)) if choose is None else choose(mae)
                    w = np.zeros(len(self.models))
                    w[sel] = 1.0 / mae[sel] ** power
                    w /= w.sum()
                    out[m_cur] = self.W[m_cur] @ w
                    log.append({"origin": int(o), "category": c, "h": int(h), **dict(zip(self.models, w))})
        return out, pd.DataFrame(log)

    def frame(self, pred: np.ndarray, first: int, name: str, ref: pd.DataFrame) -> pd.DataFrame:
        m = self.origin >= first
        res = self.keys.loc[m, ["s", "origin", "h"]].copy()
        res["yhat"] = pred[m]
        res["model"] = name
        idx = pd.MultiIndex.from_frame(res[["s", "origin", "h"]])
        r = ref.drop_duplicates(["s", "origin", "h"]).set_index(["s", "origin", "h"])
        for col in ["y", "y_origin", "scale", "target_period", "origin_period", "category"]:
            res[col] = r[col].reindex(idx).to_numpy()
        return res


def _has_both(fc: pd.DataFrame, models: list[str], first: int) -> list[str]:
    val, test = set(fc.model[fc.origin < first]), set(fc.model[fc.origin >= first])
    return [m for m in models if m in val and m in test]


def build_ensembles(fc: pd.DataFrame, members: list[str], candidates: list[str], first: int, k: int = 4):
    """Итоговый ансамбль + варианты для анализа чувствительности. Возвращает (прогнозы, сводка)."""
    out, info = [], {}
    mem = _has_both(fc, members, first)
    if len(mem) >= 2:
        p = _Panel(fc, mem)
        pred, log = p.combine(first)
        out.append(p.frame(pred, first, "Ансамбль", fc))
        info["final"] = {"members": mem, "mean_weights": {m: round(float(log[m].mean()), 3) for m in mem}}
    cand = _has_both(fc, candidates, first)
    if len(cand) >= 2:
        p = _Panel(fc, cand)
        val = (p.origin < first) & (p.target <= first)
        val_mae = p.ae[val].mean(axis=0)
        top = np.argsort(val_mae)[:k]
        variants = {
            "Э1: топ-4 по валидации 2023 г.": dict(choose=lambda mae, t=top: t),
            "Э2: все кандидаты, 1/MAE": dict(choose=None),
            "Э3: онлайн-выбор топ-4": dict(choose=lambda mae: np.argsort(mae)[:k]),
        }
        for name, kw in variants.items():
            pred, _ = p.combine(first, **kw)
            out.append(p.frame(pred, first, name, fc))
        out.append(p.frame(np.median(p.W, axis=1), first, "Э4: медиана кандидатов", fc))
        info["validation_mae"] = {m: round(float(v), 2) for m, v in zip(cand, val_mae)}
        info["validation_top"] = [cand[i] for i in top]
    return (pd.concat(out, ignore_index=True) if out else None), info
