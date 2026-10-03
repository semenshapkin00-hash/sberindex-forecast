"""Модели прогнозирования. Реестр `build_models` создаёт модели по конфигурации."""
from __future__ import annotations


def build_models(cfg, only: list[str] | None = None):
    from .baselines import (NationalSeasonalDrift, Naive, ProphetModel, SeasonalNaive, SeasonalNaiveGrowth,
                            StatsModel)
    m = cfg["models"]
    nj = cfg["project"]["n_jobs"]
    reg = {
        "naive": lambda: Naive(),
        "seasonal_naive": lambda: SeasonalNaive(),
        "seasonal_naive_growth": lambda: SeasonalNaiveGrowth(),
        "nat_seasonal": lambda: NationalSeasonalDrift(),
        "prophet": lambda: ProphetModel(m["prophet"], n_jobs=nj),
        "ets": lambda: StatsModel("ETS", False, nj),
        "ets_sa": lambda: StatsModel("ETS", True, nj),
        "theta": lambda: StatsModel("Theta", False, nj),
        "theta_sa": lambda: StatsModel("Theta", True, nj),
        "arima_sa": lambda: StatsModel("ARIMA", True, nj),
    }
    try:
        from .gbm import GlobalGBM
        reg["lightgbm"] = lambda: GlobalGBM(m["lightgbm"], use_news=True)
        reg["lightgbm_nonews"] = lambda: GlobalGBM(m["lightgbm"], use_news=False, name="LightGBM (без новостей)")
        reg["lightgbm_nocpd"] = lambda: GlobalGBM(m["lightgbm"], use_news=True, name="LightGBM (без признаков CPD)",
                                                  feature_sets=[s for s in m["lightgbm"]["feature_sets"] if s != "cpd"])
        reg["lightgbm_base"] = lambda: GlobalGBM(m["lightgbm"], use_news=False, feature_sets=["base"], name="LightGBM (только ряд)")
        reg["lightgbm_fm"] = lambda: GlobalGBM(m["lightgbm"], use_news=True, name="LightGBM + Chronos-2 (стэкинг)",
                                               fm_cache=cfg.get("_fm_cache"))
    except ImportError:
        pass
    try:
        from .foundation import ChronosBolt, Chronos2
        reg["chronos_bolt"] = lambda: ChronosBolt(m["chronos_bolt"])
        reg["chronos_bolt_sa"] = lambda: ChronosBolt(m["chronos_bolt"], deseasonalize=True, name="Chronos-Bolt-SA")
        reg["chronos2"] = lambda: Chronos2(m["chronos2"], covariates=False, name="Chronos-2")
        reg["chronos2_cov"] = lambda: Chronos2(m["chronos2"], covariates=True, name="Chronos-2 + ковариаты")
        reg["chronos2_sa"] = lambda: Chronos2(m["chronos2"], covariates=False, deseasonalize=True, name="Chronos-2-SA")
        reg["chronos2_sa_cov"] = lambda: Chronos2(m["chronos2"], covariates=True, deseasonalize=True, name="Chronos-2-SA + ковариаты")
        reg["chronos2_ft"] = lambda: Chronos2(m["chronos2"], covariates=True, deseasonalize=True, finetune=True,
                                              name="Chronos-2-SA + ковариаты + LoRA")
    except ImportError:
        pass
    keys = only or [k for k in reg if m.get(k, {}).get("enabled", True)]
    return {k: reg[k]() for k in keys if k in reg}
