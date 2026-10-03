"""Шаг 5. Сравнение методов обнаружения точек структурных изменений.

    python scripts/05_cpd_benchmark.py                 # синтетика + реальные данные + раннее предупреждение
    python scripts/05_cpd_benchmark.py --parts synthetic

A. Синтетика с известной разметкой. В реальные ряды «относительного индекса» МО вносятся
   шоки трёх типов (сдвиг уровня, постепенный сдвиг за 4 мес., временный шок на 3 мес.)
   величиной 10/20/35%; 25% рядов — без шока (для оценки ложных тревог). Ряды делятся на
   валидационные (подбор параметров методов по F1) и тестовые (итоговые метрики).
   Каждый метод оценивается ретроспективно (весь ряд) и в режиме ежемесячного мониторинга:
   в месяц t метод видит только x[:t] — так измеряются задержка обнаружения и ложные тревоги.
B. Реальные данные: найденные точки изменений сопоставляются с новостными шоками в том же
   субъекте; кейсы (паводок 2024 г. в Оренбургской, Курганской, Тюменской областях и др.).
C. Раннее предупреждение: классификатор вероятности шока в следующем месяце по информации
   до текущего месяца; сравнение с новостными признаками и без них (временное разбиение).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import cpd  # noqa: E402
from sbx.config import cli_config  # noqa: E402

warnings.filterwarnings("ignore")

OFFLINE = ["pelt", "binseg", "bottomup", "window", "kernel"]
GRIDS = {
    "pelt": [{"pen_mult": v} for v in (4, 8, 12, 16, 24, 32, 48)],
    "binseg": [{"pen_mult": v} for v in (4, 8, 12, 16, 24, 32, 48)],
    "bottomup": [{"pen_mult": v} for v in (4, 8, 12, 16, 24, 32, 48)],
    "window": [{"pen_mult": v} for v in (0.25, 0.5, 1, 2, 4)],
    "kernel": [{"pen_mult": v} for v in (4, 6, 8, 12, 16, 24)],
    "cusum": [{"h": v, "k": k} for v in (6, 8, 10, 14, 20) for k in (0.5, 1.0)],
    "page_hinkley": [{"lam": v} for v in (5, 7, 10, 14, 20, 28)],
    "bocpd": [{"hazard": hz, "threshold": th} for hz in (0.01, 0.02, 0.05, 0.1) for th in (0.2, 0.3, 0.5, 0.7)],
}
NAMES = {"pelt": "PELT", "binseg": "BinSeg", "bottomup": "BottomUp", "window": "Window", "kernel": "KernelCPD (RBF)",
         "cusum": "CUSUM", "page_hinkley": "Page–Hinkley", "bocpd": "BOCPD", "chronos_pi": "Chronos-2 интервал",
         "bocpd_news": "BOCPD + новости"}


def detect_full(method: str, x: np.ndarray, p: dict) -> list[int]:
    if method in OFFLINE:
        return cpd.offline_detect(x, method, **p)
    if method == "cusum":
        return cpd.cusum(x, **p)
    if method == "page_hinkley":
        return cpd.page_hinkley(x, **p)
    if method == "bocpd":
        return cpd.bocpd_alarms(x, p["hazard"], p["threshold"])
    raise ValueError(method)


def detect_sequential(method: str, x: np.ndarray, p: dict, start: int = 6, recent: int = 2) -> list[tuple[int, int]]:
    """Ежемесячный мониторинг: список (оценка момента изменения, месяц тревоги)."""
    if method == "cusum":
        return [(a, a) for a in cpd.cusum(x, **p)]
    if method == "page_hinkley":
        return [(a, a) for a in cpd.page_hinkley(x, **p)]
    if method == "bocpd":
        return cpd.bocpd_alarm_times(x, p["hazard"], p["threshold"])
    alarms = []
    for t in range(start, len(x)):
        for c in detect_full(method, x[: t + 1], p):
            if c >= t - recent and not any(abs(c - a) <= 1 for a, _ in alarms):
                alarms.append((c, t))
    return alarms


def make_synthetic(X: np.ndarray, rng: np.random.Generator, n: int, kinds=("level", "slope", "transient"),
                   mags=(0.10, 0.20, 0.35), null_share: float = 0.25):
    idx = rng.choice(len(X), n, replace=False)
    items = []
    for i in idx:
        if rng.random() < null_share:
            items.append({"x": X[i].copy(), "tau": None, "kind": "none", "mag": 0.0, "row": int(i)})
            continue
        kind, mag = rng.choice(kinds), float(rng.choice(mags))
        tau = int(rng.integers(8, 21))
        items.append({"x": cpd.inject(X[i], kind, tau, mag, rng), "tau": tau, "kind": kind, "mag": mag, "row": int(i)})
    return items


def score(items, method, p, sequential: bool, margin: int = 1):
    tp = fp = fn = 0
    delays, fa_null, months_null, cover = [], 0, 0, []
    for it in items:
        x, tau = it["x"], it["tau"]
        true = [] if tau is None else [tau]
        if sequential:
            al = detect_sequential(method, x, p)
            pred = [c for c, _ in al]
        else:
            pred = detect_full(method, x, p)
        a, b, c = cpd.match_cps(pred, true, margin)
        tp, fp, fn = tp + a, fp + b, fn + c
        if not sequential:
            cover.append(cpd.covering(pred, true, len(x)))
        if tau is None:
            fa_null += len(pred)
            months_null += len(x) - 6
        elif sequential and a:
            t_al = min(t for cc, t in al if abs(cc - tau) <= margin)
            delays.append(t_al - tau)
    out = cpd.f1_scores(tp, fp, fn)
    out["false_alarms_per_100_months"] = 100 * fa_null / max(months_null, 1)
    if sequential:
        out["mean_delay"] = float(np.mean(delays)) if delays else np.nan
        out["detected_within_1m"] = float(np.mean(np.array(delays) <= 1)) if delays else 0.0
    else:
        out["covering"] = float(np.mean(cover))
    return out


def _score_job(items, method, p, sequential):
    return method, p, sequential, score(items, method, p, sequential)


def chronos_interval_alarms(items, cfg, levels=((0.05, 0.95), (0.025, 0.975)), start=6):
    """Детектор по выходу факта за прогнозный интервал Chronos-2 (одношаговый прогноз x_t по x[:t]).

    Возвращает {уровень: [список (момент, тревога) для каждого ряда]}.
    """
    import torch
    from chronos import Chronos2Pipeline

    from sbx.forecast.foundation import _device
    pipe = Chronos2Pipeline.from_pretrained(cfg.models["chronos2"]["model_id"], device_map=_device("auto"))
    qs = sorted({q for lv in levels for q in lv})
    jobs = [(k, t) for k, it in enumerate(items) for t in range(start, len(it["x"]))]
    Q = {}
    bs = 512
    for b in range(0, len(jobs), bs):
        chunk = jobs[b: b + bs]
        ctxs = [torch.tensor(items[k]["x"][:t], dtype=torch.float32) for k, t in chunk]
        q, _ = pipe.predict_quantiles(ctxs, prediction_length=1, quantile_levels=qs)
        for (k, t), qq in zip(chunk, q):
            Q[(k, t)] = dict(zip(qs, qq.float().cpu().numpy().reshape(-1)))
    out = {}
    for lo, hi in levels:
        res = []
        for k, it in enumerate(items):
            x = it["x"]
            ex = [t for t in range(start, len(x)) if x[t] < Q[(k, t)][lo] or x[t] > Q[(k, t)][hi]]
            al = [t for i, t in enumerate(ex) if i == 0 or t - ex[i - 1] > 1]   # серия выходов = одно событие
            res.append([(t, t) for t in al])
        out[f"q{lo}–q{hi}"] = res
    return out


def two_stage(signal: list[tuple[int, int]], confirm: list[tuple[int, int]], window: int = 1) -> list[tuple[int, int]]:
    """Сигнал (чувствительный детектор) подтверждается вторым методом в пределах window месяцев."""
    out = []
    for c, t in signal:
        hits = [tc for cc, tc in confirm if abs(cc - c) <= window and tc <= t + window]
        if hits:
            out.append((c, max(t, min(hits))))
    return out


def score_alarms(items, alarms, margin=1):
    tp = fp = fn = 0
    delays, fa, mon = [], 0, 0
    for it, al in zip(items, alarms):
        true = [] if it["tau"] is None else [it["tau"]]
        pred = [c for c, _ in al]
        a, b, c = cpd.match_cps(pred, true, margin)
        tp, fp, fn = tp + a, fp + b, fn + c
        if it["tau"] is None:
            fa, mon = fa + len(pred), mon + len(it["x"]) - 6
        elif a:
            delays.append(min(t for cc, t in al if abs(cc - it["tau"]) <= margin) - it["tau"])
    s = cpd.f1_scores(tp, fp, fn)
    return {**s, "false_alarms_per_100_months": 100 * fa / max(mon, 1), "mean_delay": float(np.mean(delays)) if delays else np.nan,
            "detected_within_1m": float(np.mean(np.array(delays) <= 1)) if delays else 0.0}


def part_synthetic(cfg, X, out: Path):
    rng = np.random.default_rng(cfg.project["seed"])
    nval, ntest = 1200, 3000
    val = make_synthetic(X, rng, nval)
    test = make_synthetic(X, np.random.default_rng(cfg.project["seed"] + 1), ntest)
    nj = cfg.project["n_jobs"]
    # подбор параметров на валидации (по F1 в режиме мониторинга — цель: раннее и точное обнаружение)
    jobs = [(m, p) for m, grid in GRIDS.items() for p in grid]
    best = {}
    res = Parallel(n_jobs=nj)(delayed(_score_job)(val, m, p, True) for m, p in jobs)
    tune = pd.DataFrame([{"method": m, "params": json.dumps(p), **s} for m, p, _, s in res])
    tune.to_csv(out / "cpd_tuning.csv", index=False)
    for m in GRIDS:
        sub = tune[tune.method == m].sort_values("F1", ascending=False)
        best[m] = json.loads(sub.iloc[0].params)
    print("лучшие параметры:", best)
    # итоговая оценка на тесте
    rows = []
    res = Parallel(n_jobs=nj)(delayed(_score_job)(test, m, best[m], seq) for m in GRIDS for seq in (False, True))
    for m, p, seq, s in res:
        rows.append({"method": NAMES[m], "mode": "мониторинг" if seq else "ретроспектива", "params": json.dumps(p), **s})
    # Chronos-2: выход за прогнозный интервал и двухуровневые схемы «сигнал → подтверждение»
    try:
        ch = chronos_interval_alarms(test, cfg)
        for lv, al in ch.items():
            rows.append({"method": NAMES["chronos_pi"], "mode": "мониторинг", "params": lv, **score_alarms(test, al)})
        sig = ch["q0.025–q0.975"]
        for conf in ("pelt", "bocpd", "binseg"):
            confirm = [detect_sequential(conf, it["x"], best[conf]) for it in test]
            rows.append({"method": f"Chronos-2 → {NAMES[conf]}", "mode": "мониторинг", "params": "q0.025–q0.975, окно 1",
                         **score_alarms(test, [two_stage(s, c) for s, c in zip(sig, confirm)])})
    except Exception as e:  # модель недоступна — сравнение без неё
        print("Chronos-2 интервал пропущен:", e)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "cpd_synthetic_summary.csv", index=False)
    # разрез по типу и величине шока (мониторинг)
    det = []
    for m in GRIDS:
        for kind in ("level", "slope", "transient"):
            for mag in (0.10, 0.20, 0.35):
                sub = [it for it in test if it["kind"] == kind and it["mag"] == mag]
                s = score(sub, m, best[m], True)
                det.append({"method": NAMES[m], "kind": kind, "mag": mag, "recall": s["recall"], "mean_delay": s["mean_delay"]})
    pd.DataFrame(det).to_csv(out / "cpd_synthetic_by_type.csv", index=False)
    (out / "cpd_best_params.json").write_text(json.dumps(best, indent=1))
    print(summ.round(3).to_string())
    return best


def main():
    cfg, args = cli_config(__doc__, lambda ap: ap.add_argument("--parts", nargs="*", default=["synthetic", "real"]))
    proc = cfg.path("processed")
    out = cfg.path("results") / "cpd"
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(proc / "panel.parquet")
    W = panel.pivot_table(index=["territory_id", "category"], columns="period", values="value")
    W = W[W.notna().sum(axis=1) == 24]
    meta = W.index.to_frame(index=False).merge(pd.read_parquet(proc / "static.parquet"), on="territory_id", how="left")
    X = cpd.relative_index(W.to_numpy(), cpd.peer_groups(meta))
    best = None
    if "synthetic" in args.parts:
        best = part_synthetic(cfg, X, out)
    if "real" in args.parts:
        from sbx.cpd_real import run_real
        if best is None:
            best = json.loads((out / "cpd_best_params.json").read_text())
        run_real(cfg, W, X, best, out)


if __name__ == "__main__":
    main()
