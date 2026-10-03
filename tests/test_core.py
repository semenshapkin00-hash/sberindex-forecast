"""Минимальные проверки ключевых свойств конвейера (запуск: .venv/bin/python -m pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbx import cpd, metrics  # noqa: E402
from sbx.forecast.base import Context  # noqa: E402
from sbx.forecast.baselines import Naive, SeasonalNaive  # noqa: E402


def _ctx(S=5, T=24, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.exp(10 + rng.normal(0, 0.05, (S, T)).cumsum(axis=1))
    periods = [str(p) for p in pd.period_range("2023-01", periods=T, freq="M")]
    meta = pd.DataFrame({"territory_id": range(S), "category": ["Все категории"] * S, "region_key": ["r"] * S})
    empty = pd.DataFrame({"period": periods})
    return Context(Y, periods, meta, empty, empty, pd.DataFrame())


def test_masked_hides_future():
    ctx = _ctx()
    m = ctx.masked(10)
    assert np.isnan(m.Y[:, 11:]).all() and np.isfinite(m.Y[:, :11]).all()
    assert np.isfinite(ctx.Y).all(), "исходная матрица не должна меняться"


def test_baselines_use_only_history():
    ctx = _ctx()
    o = 15
    f_full = SeasonalNaive().predict(ctx.masked(o), o, 6)
    ctx2 = _ctx()
    ctx2.Y[:, o + 1:] *= 100       # «будущее» меняется — прогноз не должен
    f_changed = SeasonalNaive().predict(ctx2.masked(o), o, 6)
    assert np.allclose(f_full, f_changed)
    assert np.allclose(Naive().predict(ctx.masked(o), o, 3), ctx.Y[:, [o]])


def test_metrics():
    y, f = np.array([100, 200, 300.0]), np.array([110, 190, 300.0])
    assert metrics.mae(y, f) == 20 / 3
    assert metrics.r2(y, y) == 1.0
    res = metrics.paired_test(np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 4, 5]))
    assert res["diff"] == -1 and res["share_better"] == 1


def test_detectors_find_obvious_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, 24)
    x[14:] += 0.4
    for m in ("pelt", "binseg", "kernel"):
        cps = cpd.offline_detect(x, m, pen_mult=8)
        assert any(abs(c - 14) <= 1 for c in cps), m
    assert any(abs(c - 14) <= 1 for c, _ in cpd.bocpd_alarm_times(x, 0.05, 0.3))
    assert any(abs(c - 14) <= 1 for c in cpd.cusum(x, h=6, k=1.0))


def test_cpd_matching_and_covering():
    assert cpd.match_cps([5, 12], [12], margin=1) == (1, 1, 0)
    assert cpd.covering([10], [10], 24) == 1.0


def test_news_lemmas_and_regions():
    from sbx import news
    assert news.lemma("Курской") == "курский"
    assert news.lemmatize("В Орске паводок")[1] == "орск"
