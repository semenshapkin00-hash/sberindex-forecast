"""Шаг 8. Интерактивный лендинг: шаблон landing/template.html + данные результатов.

    python scripts/08_build_landing.py [--repo https://github.com/<user>/<repo>]

Выход: docs/index.html (самостоятельная страница для GitHub Pages) и landing/artifact.html
(та же страница без обёртки документа — для публикации как Artifact).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import cpd  # noqa: E402
from sbx.config import ROOT, cli_config  # noqa: E402

SHOW = ["Ансамбль", "LightGBM (только ряд)", "LightGBM", "LightGBM (без новостей)", "LightGBM + Chronos-2 (стэкинг)", "SNaive×Growth",
        "NatSeasonal", "Theta-SA", "Chronos-2-SA + ковариаты", "Chronos-2", "Chronos-Bolt", "Prophet", "Naive"]
FM = ["Chronos-Bolt", "Chronos-Bolt-SA", "Chronos-2", "Chronos-2-SA", "Chronos-2 + ковариаты", "Chronos-2-SA + ковариаты",
      "Chronos-2-SA + ковариаты + LoRA", "LightGBM + Chronos-2 (стэкинг)", "SNaive×Growth", "Prophet"]
r = lambda x, d=4: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def build(cfg, repo: str | None) -> dict:
    res, cdir, proc = cfg.path("results") / "full", cfg.path("results") / "cpd", cfg.path("processed")
    m = pd.read_csv(res / "metrics_by_window.csv")
    final = "Ансамбль" if "Ансамбль" in set(m.model) else m[m.window == 12].sort_values("MAE").model.iloc[0]
    piv = {k: m.pivot(index="model", columns="window", values=c) for k, c in [("MAE", "MAE"), ("R2dyn", "R2_dyn"), ("WAPE", "WAPE")]}
    models = [x for x in SHOW if x in piv["MAE"].index]
    models = sorted(models, key=lambda x: piv["MAE"].loc[x].mean())
    D = {"meta": {"final": final, "generated": dt.date.today().strftime("%d.%m.%Y"), "repo": repo}}
    D["kpi"] = {"gain": {H: r(1 - piv["MAE"].loc[final, H] / piv["MAE"].loc["Prophet", H]) for H in (1, 3, 6, 12)},
                "final_mae": {H: r(piv["MAE"].loc[final, H], 1) for H in (1, 3, 6, 12)},
                "prophet_mae": {H: r(piv["MAE"].loc["Prophet", H], 1) for H in (1, 3, 6, 12)}}
    D["windows"] = {"models": models, **{k: {x: [r(v.loc[x, H]) for H in (1, 3, 6, 12)] for x in models} for k, v in piv.items()}}
    s = pd.read_csv(res / "metrics_by_step.csv")
    sm = [x for x in [final, "LightGBM", "SNaive×Growth", "Chronos-2-SA + ковариаты", "Prophet"] if x in set(s.model)]
    sm = list(dict.fromkeys(sm))
    D["steps"] = {"models": sm, "h": list(range(1, 13)),
                  "MAE": {x: [r(v, 1) for v in s[s.model == x].sort_values("h").MAE] for x in sm}}
    mc = pd.read_csv(res / "metrics_by_window_category.csv")
    a = mc[mc.model == final].set_index(["category", "window"]).MAE
    b = mc[mc.model == "Prophet"].set_index(["category", "window"]).MAE
    g = (1 - a / b).unstack("window")
    cats = ["Все категории", "Продовольствие", "Маркетплейсы", "Общественное питание", "Транспорт", "Здоровье"]
    g = g.loc[[c for c in cats if c in g.index]]
    D["cat_gain"] = {"cats": list(g.index), "windows": [1, 3, 6, 12], "z": [[r(v) for v in row] for row in g.to_numpy()]}
    t = pd.read_csv(res / "tests_vs_prophet.csv")
    t = t[t.model == final]
    D["tests"] = [{"window": int(x.window), "diff": r(x.diff, 2), "lo": r(x.ci_low, 2), "hi": r(x.ci_high, 2),
                   "p": float(x.p_wilcoxon), "share": r(x.share_better)} for x in t.itertuples()]
    fm = [x for x in FM if x in piv["MAE"].index]
    D["foundation"] = {"models": fm, "MAE": {x: [r(piv["MAE"].loc[x, H], 1) for H in (1, 3, 6, 12)] for x in fm}}
    cs = pd.read_csv(cdir / "cpd_synthetic_summary.csv")
    cs["label"] = cs.method + np.where(cs.method == "Chronos-2 интервал", " " + cs.params.str.replace("q", "").str.replace("–", "–"), "")
    D["cpd"] = [{"label": x.label, "mode": x.mode, "precision": r(x.precision), "recall": r(x.recall), "F1": r(x.F1),
                 "fa": r(x.false_alarms_per_100_months), "delay": r(x.mean_delay)} for x in cs.itertuples()]
    bt = pd.read_csv(cdir / "cpd_synthetic_by_type.csv")
    D["cpd_type"] = [{"method": x.method, "kind": x.kind, "mag": x.mag, "recall": r(x.recall)} for x in bt.itertuples()]
    es = pd.read_csv(cdir / "news_event_study.csv")
    D["events"] = [{"type": x.type, "category": x.category, "k": int(x.k), "mean": r(x.mean, 3), "lo": r(x.ci_low, 3),
                    "hi": r(x.ci_high, 3)} for x in es.itertuples()]
    # новости
    nw = pd.read_parquet(proc / "news_region_monthly.parquet")
    st = pd.read_parquet(proc / "static.parquet")
    names = st.drop_duplicates("region_key").set_index("region_key").region_name
    reg = nw[nw.territory_id.isna() & (nw.region_key != "__RU__") & (nw.period >= "2023-01")]
    top = reg.groupby("region_key").news_shock_surprise.max().sort_values(ascending=False).head(16).index
    hp = reg[reg.region_key.isin(top)].pivot_table(index="region_key", columns="period", values="news_shock_surprise").loc[top].clip(lower=0)
    D["news_heat"] = {"regions": [names.get(k, k) for k in top], "periods": list(hp.columns), "z": [[r(v, 2) for v in row] for row in hp.to_numpy()]}
    tg = pd.read_parquet(proc / "news_tagged.parquet", columns=["regions", "mos"])
    D["meta"].update({"n_headlines": int(len(tg)), "geo_share": f"{(tg.regions.str.len() > 0).mean():.1%}".replace(".", ","),
                      "mo_share": f"{(tg.mos.str.len() > 0).mean():.1%}".replace(".", ",")})
    # кейсы
    cases = json.loads((cdir / "real_cases.json").read_text())
    orsk = [c for c in cases if c["mo"] == "городской округ город Орск"]
    D["cases"] = {"orsk": {"periods": orsk[0]["periods"],
                           "x": {c["category"]: [r(v) for v in c["x"]] for c in orsk if c["category"] in ("Все категории", "Маркетплейсы", "Здоровье", "Общественное питание", "Продовольствие")},
                           "bocpd": {c["category"]: [tt for _, tt in c.get("bocpd", [])] for c in orsk}}}
    per = orsk[0]["periods"]
    ev = per.index("2024-04")
    fl = {}
    for c in cases:
        if c["category"] == "Маркетплейсы":
            x = np.array(c["x"]) * 100
            x = x - x[ev - 3: ev].mean()
            fl[c["case"].split(" (")[0]] = [r(v, 2) for v in x[ev - 6: ev + 6]]
    D["cases"]["flood"] = {"periods": per[ev - 6: ev + 6], "x": fl}
    # апрель 2023
    p = pd.read_parquet(proc / "panel.parquet").merge(st[["territory_id", "region_name", "mo_type"]], on="territory_id")
    fed = p.region_name.isin(["Москва", "Санкт-Петербург"])
    p["grp"] = np.where(fed, "города фед. значения", np.where(p.mo_type == 0, "городские округа", "районы и мун. округа"))
    D["april"] = {"periods": [], "series": {}}
    for cat in ["Здоровье", "Общественное питание", "Транспорт", "Продовольствие"]:
        dd = p[(p.category == cat) & (p.period <= "2023-12")].pivot_table(index="grp", columns="period", values="value", aggfunc="median")
        dd = np.log(dd.div(dd["2023-03"], axis=0)) * 100
        D["april"]["periods"] = list(dd.columns)
        D["april"]["series"][cat] = {gname: [r(v, 2) for v in dd.loc[gname]] for gname in dd.index}
    # проводник
    D["explorer"] = explorer(cfg, final, st)
    D["meta"]["n_mo"] = len(D["explorer"]["ids"])
    full = pd.read_parquet(proc / "panel.parquet").groupby(["territory_id", "category"]).size()
    D["meta"]["n_series"] = int((full >= cfg.data["min_obs"]).sum())
    return D


def explorer(cfg, final: str, st: pd.DataFrame) -> dict:
    res, cdir, proc = cfg.path("results") / "full", cfg.path("results") / "cpd", cfg.path("processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    periods = sorted(panel.period.unique())
    cats = ["Все категории", "Продовольствие", "Маркетплейсы", "Общественное питание", "Транспорт", "Здоровье"]
    st = st.set_index("territory_id")
    ids = sorted(panel.territory_id.unique())
    W = panel.pivot_table(index=["territory_id", "category"], columns="period", values="value").reindex(columns=periods)
    fed = st.region_name.isin(["Москва", "Санкт-Петербург", "Севастополь"])
    kind = np.where(fed, "fed", np.where(st.mo_type == 0, "urban", "rural"))
    kind = pd.Series(kind, index=st.index)
    y = []
    for tid in ids:
        row = []
        for c in cats:
            if (tid, c) in W.index:
                v = W.loc[(tid, c)].to_numpy()
                row.append([None if np.isnan(x) else int(x) for x in v])
            else:
                row.append(None)
        y.append(row)
    # прогнозы из декабря 2023 г.
    fc = pd.read_parquet(res / "all_forecasts.parquet", columns=["model", "s", "origin_period", "h", "yhat"])
    fc = fc[(fc.origin_period == "2023-12") & fc.model.isin([final, "Prophet"])]
    from sbx.forecast.base import build_context
    ctx = build_context(proc, cfg.path("external"), cfg)
    key = ctx.meta[["territory_id", "category"]].reset_index().rename(columns={"index": "s"})
    fc = fc.merge(key, on="s")
    out = {}
    for mname, part in fc.groupby("model"):
        P = part.pivot_table(index=["territory_id", "category"], columns="h", values="yhat")
        arr = []
        for tid in ids:
            arr.append([[int(v) for v in P.loc[(tid, c)].to_numpy()] if (tid, c) in P.index else None for c in cats])
        out[mname] = arr
    cb = pd.read_csv(cdir / "real_consensus_breaks.csv")
    cb = cb[cb.magnitude.abs() >= 0.10]
    cps = {(a, b): sorted(g.cp.astype(int).tolist()) for (a, b), g in cb.groupby(["territory_id", "category"])}
    nw = pd.read_parquet(proc / "news_region_monthly.parquet")
    mo = nw[nw.territory_id.notna() & (nw.news_shock_mo > 0)]
    news = {}
    for x in mo.itertuples():
        if x.period in periods:
            news.setdefault(int(x.territory_id), []).append([periods.index(x.period), int(round(np.expm1(x.news_shock_mo)))])
    pop = st.population.reindex(ids)
    return {"periods": periods, "cats": cats, "ids": [int(i) for i in ids],
            "names": st.mo_name.reindex(ids).tolist(), "regions": st.region_name.reindex(ids).tolist(),
            "kind": kind.reindex(ids).tolist(), "lat": [r(v, 3) for v in st.lat.reindex(ids)], "lon": [r(v, 3) for v in st.lon.reindex(ids)],
            "pop": [None if np.isnan(v) else int(v) for v in pop], "y": y,
            "f_final": out.get(final, [None] * len(ids)), "f_prophet": out.get("Prophet", [None] * len(ids)),
            "cps": [[cps.get((tid, c), []) for c in cats] for tid in ids], "news": [news.get(int(tid), []) for tid in ids]}


def main():
    cfg, args = cli_config(__doc__, lambda ap: ap.add_argument("--repo", default=None))
    D = build(cfg, args.repo)
    tpl = (ROOT / "landing" / "template.html").read_text()
    page = tpl.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
    (ROOT / "landing" / "artifact.html").write_text(page)
    head_end = page.index("</style>") + len("</style>")
    full = ("<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + page[:head_end] + "\n</head>\n<body>\n" + page[head_end:] + "\n</body>\n</html>\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(full)
    print(f"лендинг: docs/index.html ({len(full) / 1e6:.1f} МБ)")
    # PDF-версия презентации: печать страницы headless-браузером (стили @media print)
    import shutil
    import subprocess
    chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]
                   if Path(c).exists() or shutil.which(c)), None)
    if chrome:
        pdf = ROOT / "reports" / "presentation.pdf"
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                        "--blink-settings=preferredColorScheme=1", f"--print-to-pdf={pdf}", (ROOT / "docs" / "index.html").as_uri()],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("PDF-версия:", pdf)


if __name__ == "__main__":
    main()
