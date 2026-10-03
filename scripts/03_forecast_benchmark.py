"""Шаг 3. Бэктест моделей прогнозирования и сравнение с Prophet.

    python scripts/03_forecast_benchmark.py                       # все включённые модели
    python scripts/03_forecast_benchmark.py --models prophet seasonal_naive_growth
    python scripts/03_forecast_benchmark.py --set backtest.sample_series=500   # быстрый прогон
    python scripts/03_forecast_benchmark.py --evaluate-only       # пересчитать таблицы

Прогнозы каждой модели кэшируются в results/forecasts/<model>.parquet; повторный запуск
пересчитывает только модели без кэша (или с флагом --force).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import backtest, metrics  # noqa: E402
from sbx.config import cli_config  # noqa: E402
from sbx.forecast import build_models  # noqa: E402
from sbx.forecast.base import build_context  # noqa: E402
from sbx.forecast.ensemble import build_ensembles  # noqa: E402


def args_ext(ap):
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--warmup", action="store_true",
                    help="прогнать модели из валидационных точек (до first_origin) для выбора участников ансамбля")


def main():
    cfg, args = cli_config(__doc__, args_ext)
    np.random.seed(cfg.project["seed"])
    ctx = build_context(cfg.path("processed"), cfg.path("external"), cfg, cfg.data["min_obs"])
    n = cfg.backtest.get("sample_series")
    tag = "full"
    if n:
        idx = np.sort(np.random.default_rng(cfg.project["seed"]).choice(ctx.S, int(n), replace=False))
        ctx.Y, ctx.meta = ctx.Y[idx], ctx.meta.iloc[idx].reset_index(drop=True)
        tag = f"sample{n}"
    out = cfg.path("results") / tag
    fdir = out / "forecasts"
    org = backtest.origins(ctx, str(cfg.backtest["first_origin"]), str(cfg.backtest["last_origin"]))
    first = org[0]
    if args.warmup:   # валидационные точки: только для выбора и начальных весов ансамбля
        org = backtest.origins(ctx, str(cfg.backtest["warmup_first_origin"]), ctx.periods[first - 1])
        fdir = fdir / "warmup"
    print(f"рядов: {ctx.S:,}; точек прогноза: {len(org)} ({ctx.periods[org[0]]} … {ctx.periods[org[-1]]})")

    if not args.evaluate_only:
        wanted = args.models or [k for k, v in cfg.models.items() if isinstance(v, dict) and v.get("enabled", True)]
        if "lightgbm_fm" in wanted:
            # признаки стэкинга: прогнозы Chronos-2 из каждой исторической точки (кэш общий для всех прогонов)
            from sbx.forecast.foundation import precompute_fm_features
            full = cfg.path("results") / "full" / "fm_features.parquet"
            path = full if full.exists() or tag == "full" else out / "fm_features.parquet"
            base = out / "forecasts"
            reuse = [base / "chronos2_sa_cov.parquet", base / "warmup" / "chronos2_sa_cov.parquet"]
            precompute_fm_features(ctx, cfg.models["chronos2"], path, reuse=reuse if all(p.exists() for p in reuse) else None)
            cfg["_fm_cache"] = str(path)
        for key, model in build_models(cfg, args.models).items():
            path = fdir / f"{key}.parquet"
            if path.exists() and not args.force:
                continue
            print(f"▶ {model.name}")
            fc = backtest.run_model(model, ctx, org, cfg.backtest["max_horizon"])
            fc["model"] = model.name
            imp = fc.attrs.pop("importance", None)
            if imp is not None and not args.warmup:
                imp.to_csv(out / f"importance_{key}.csv")
            backtest.save_forecasts(fc, path)

    if args.warmup:
        return
    fdir = out / "forecasts"
    frames = [pd.read_parquet(p) for p in sorted(fdir.glob("*.parquet"))]
    frames += [pd.read_parquet(p) for p in sorted((fdir / "warmup").glob("*.parquet"))]
    fc = backtest.attach_truth(pd.concat(frames, ignore_index=True), ctx)
    if cfg.models["ensemble"]["enabled"]:
        e = cfg.models["ensemble"]
        ens, info = build_ensembles(fc, e["members"], e["candidates"], first, e.get("k", 4))
        if ens is not None:
            fc = pd.concat([fc, ens], ignore_index=True)
            (out / "ensemble_selection.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
            print("ансамбль:", info)
    fc = fc[fc.origin >= first].dropna(subset=["yhat"])   # тестовые метрики — только с первой тестовой точки
    res = backtest.evaluate(fc, T=len(ctx.periods))
    for k, v in res.items():
        v.to_csv(out / f"metrics_{k}.csv", index=False)
    fc.to_parquet(out / "all_forecasts.parquet")

    # парное сравнение каждой модели с Prophet (ошибки усреднены внутри ряда)
    tests = []
    w = backtest.window_view(fc, T=len(ctx.periods))
    w["ae"] = (w.y - w.yhat).abs()
    per = w.groupby(["model", "window", "s"]).ae.mean().unstack("model")
    if "Prophet" in per:
        for m in per.columns:
            if m == "Prophet":
                continue
            for H in sorted(w.window.unique()):
                sub = per.xs(H, level="window")[[m, "Prophet"]].dropna()
                tests.append({"model": m, "window": H, **metrics.paired_test(sub[m].to_numpy(), sub["Prophet"].to_numpy())})
    pd.DataFrame(tests).to_csv(out / "tests_vs_prophet.csv", index=False)

    tab = res["by_window"].pivot(index="model", columns="window", values="MAE").sort_values(12)
    print("\nMAE по окнам прогноза (руб.):")
    print(tab.round(1).to_string())
    print("\nR² по окнам прогноза:")
    print(res["by_window"].pivot(index="model", columns="window", values="R2").loc[tab.index].round(4).to_string())
    (out / "summary.json").write_text(json.dumps({"n_series": int(ctx.S), "mae": tab.round(2).to_dict()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
