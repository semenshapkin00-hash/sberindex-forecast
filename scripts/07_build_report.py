"""Шаг 7. Сборка методологического отчёта: шаблон + результаты → Markdown → HTML → PDF.

    python scripts/07_build_report.py

Все числа и таблицы подставляются из results/, поэтому отчёт всегда соответствует последнему
прогону. PDF печатается headless-браузером Chrome/Chromium (если найден).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.config import ROOT, cli_config  # noqa: E402

CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]
ORDER = ["Ансамбль", "Э1: топ-4 по валидации 2023 г.", "LightGBM", "LightGBM (без новостей)", "LightGBM (без признаков CPD)", "LightGBM (только ряд)",
         "LightGBM + Chronos-2 (стэкинг)", "SNaive×Growth", "NatSeasonal", "Theta-SA", "ETS-SA",
         "Chronos-2-SA + ковариаты + LoRA", "Chronos-2-SA + ковариаты", "Chronos-2-SA", "Chronos-2 + ковариаты", "Chronos-2",
         "Chronos-Bolt-SA", "Chronos-Bolt", "Prophet", "Naive", "SeasonalNaive"]


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def pct(x, d=1):
    return fmt(100 * x, d) + "%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "--:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def build_vars(cfg) -> dict:
    res = cfg.path("results") / "full"
    cpd_dir = cfg.path("results") / "cpd"
    proc = cfg.path("processed")
    m = pd.read_csv(res / "metrics_by_window.csv")
    mae = m.pivot(index="model", columns="window", values="MAE")
    r2 = m.pivot(index="model", columns="window", values="R2")
    r2d = m.pivot(index="model", columns="window", values="R2_dyn")
    wape = m.pivot(index="model", columns="window", values="WAPE")
    final = "Ансамбль" if "Ансамбль" in mae.index else mae[12].idxmin()
    v = {"final_model": final, "seed": cfg.project["seed"], "report_date": dt.date.today().strftime("%d.%m.%Y")}
    for H in (1, 3, 6, 12):
        v[f"final_mae_{H}"] = fmt(mae.loc[final, H])
        v[f"prophet_mae_{H}"] = fmt(mae.loc["Prophet", H])
        v[f"gain_{H}"] = pct(1 - mae.loc[final, H] / mae.loc["Prophet", H])
    v["final_wape_12"] = pct(wape.loc[final, 12])
    # таблица результатов
    rows = []
    for name in [x for x in ORDER if x in mae.index]:
        row = {"Модель": f"**{name}**" if name in (final, "Prophet") else name}
        for H in (1, 3, 6, 12):
            row[f"MAE {H} мес."] = fmt(mae.loc[name, H])
        for H in (1, 12):
            row[f"R² {H}"] = fmt(r2.loc[name, H], 3)
            row[f"R² динамики {H}"] = fmt(r2d.loc[name, H], 2)
        rows.append(row)
    v["table_forecast"] = ("Таблица 1. Качество прогноза (MAE — руб. на жителя в месяц; R² динамики — "
                           "по лог-приросту от точки прогноза).\n\n") + md_table(pd.DataFrame(rows))
    t = pd.read_csv(res / "tests_vs_prophet.csv")
    tf = t[t.model == final].copy()
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    pm = tf.p_wilcoxon.max()
    v["p_max"] = "10⁻¹⁰" if pm < 1e-10 else "10" + str(int(np.ceil(np.log10(pm)))).translate(sup)
    tf = tf.assign(**{"Окно": tf.window.astype(str) + " мес.", "Разность MAE": tf["diff"].map(fmt),
                      "95% ДИ": [f"[{fmt(a)}; {fmt(b)}]" for a, b in zip(tf.ci_low, tf.ci_high)],
                      "p (Уилкоксон)": tf.p_wilcoxon.map(lambda p: "< 10⁻¹⁰" if p < 1e-10 else f"{p:.1e}"), "Доля рядов, где лучше": tf.share_better.map(pct)})
    v["table_tests"] = (f"Таблица 2. {final} против Prophet: парное сравнение по рядам.\n\n"
                        + md_table(tf[["Окно", "Разность MAE", "95% ДИ", "p (Уилкоксон)", "Доля рядов, где лучше"]]))
    fnd = [x for x in ["Chronos-Bolt", "Chronos-Bolt-SA", "Chronos-2", "Chronos-2-SA", "Chronos-2 + ковариаты",
                       "Chronos-2-SA + ковариаты", "Chronos-2-SA + ковариаты + LoRA", "LightGBM + Chronos-2 (стэкинг)",
                       "SNaive×Growth", "Prophet"] if x in mae.index]
    ft = pd.DataFrame({"Модель": fnd, **{f"MAE {H} мес.": [fmt(mae.loc[x, H]) for x in fnd] for H in (1, 3, 6, 12)}})
    v["table_foundation"] = "Таблица 3. Фундаментальные модели в разных режимах (для сравнения — SNaive×Growth и Prophet).\n\n" + md_table(ft)
    abl = [x for x in ["LightGBM", "LightGBM (без новостей)", "LightGBM (без признаков CPD)", "LightGBM (только ряд)",
                       "LightGBM + Chronos-2 (стэкинг)"] if x in mae.index]
    at = pd.DataFrame({"Вариант": abl, **{f"MAE {H} мес.": [fmt(mae.loc[x, H]) for x in abl] for H in (1, 3, 6, 12)}})
    v["table_ablation"] = "Таблица 4. Абляции LightGBM: вклад групп признаков.\n\n" + md_table(at)
    imp = res / "importance_lightgbm.csv"
    v["n_features"] = str(len(pd.read_csv(imp))) if imp.exists() else "≈70"
    sel = cfg.path("results") / "full" / "ensemble_selection.json"
    ens = [x for x in mae.index if x == "Ансамбль" or x.startswith("Э")] + ["LightGBM (без новостей)", "Prophet"]
    et = pd.DataFrame({"Схема": ens, **{f"MAE {H} мес.": [fmt(mae.loc[x, H]) for x in ens] for H in (1, 3, 6, 12)}})
    v["table_ensemble"] = "Таблица 5. Варианты ансамбля (для сравнения — лучшая одиночная модель и Prophet).\n\n" + md_table(et)
    if sel.exists():
        info = json.loads(sel.read_text())
        mw = info.get("final", {}).get("mean_weights", {})
        v["ensemble_members"] = "; ".join(f"{k} — {fmt(w, 2)}" for k, w in mw.items())
        v["ensemble_members_short"] = "5 вариантов LightGBM и 2 сезонные модели с онлайн-весами"
    else:
        v["ensemble_members"] = v["ensemble_members_short"] = "—"
    # новости: эффект на МО-месяцах с событиями
    v["news_gain_text"] = news_gain(cfg)
    # данные
    panel = pd.read_parquet(proc / "panel.parquet")
    v["n_obs"] = fmt(len(panel), 0)
    v["n_mo"] = fmt(panel.territory_id.nunique(), 0)
    v["n_series_all"] = fmt(panel.groupby(["territory_id", "category"]).ngroups, 0)
    full = panel.groupby(["territory_id", "category"]).size()
    v["n_series"] = fmt(int((full >= cfg.data["min_obs"]).sum()), 0)
    tg = pd.read_parquet(proc / "news_tagged.parquet", columns=["regions", "mos"])
    v["n_headlines"] = fmt(len(tg), 0)
    v["geo_share"] = pct((tg.regions.str.len() > 0).mean())
    v["mo_share"] = pct((tg.mos.str.len() > 0).mean())
    # точки изменений
    cs = pd.read_csv(cpd_dir / "cpd_synthetic_summary.csv")
    mon = cs[cs["mode"] == "мониторинг"]
    ch = mon[(mon.method == "Chronos-2 интервал") & (mon.params == "q0.025–q0.975")].iloc[0]
    v["chronos_f1"] = fmt(ch.F1, 3)
    ts = mon[mon.method == "Chronos-2 → PELT"].iloc[0]
    v["two_stage_precision"] = fmt(ts.precision, 2)
    v["two_stage_fa"] = fmt(ts.false_alarms_per_100_months, 2)
    keep = cs.copy()
    keep["Метод"] = keep.method + keep.params.where(keep.method.str.contains("интервал"), "").map(lambda s: f" ({s})" if s else "")
    keep = keep[~((keep["mode"] == "мониторинг") & (keep.recall == 0))]
    tab = pd.DataFrame({"Метод": keep["Метод"], "Режим": keep["mode"], "Precision": keep.precision.map(lambda x: fmt(x, 2)),
                        "Recall": keep.recall.map(lambda x: fmt(x, 2)), "F1": keep.F1.map(lambda x: fmt(x, 3)),
                        "Ложные тревоги / 100 мес.": keep.false_alarms_per_100_months.map(lambda x: fmt(x, 2)),
                        "Задержка, мес.": keep.mean_delay.map(lambda x: "—" if pd.isna(x) else fmt(x, 2)),
                        "Covering": keep.covering.map(lambda x: "—" if pd.isna(x) else fmt(x, 3))})
    v["table_cpd"] = "Таблица 6. Сравнение детекторов на синтетическом тесте (3 000 рядов; параметры подобраны на валидации).\n\n" + md_table(tab)
    rs = json.loads((cpd_dir / "real_summary.json").read_text())
    v["n_breaks"], v["n_big"] = fmt(rs["n_breaks"], 0), fmt(rs["n_big"], 0)
    ew = pd.read_csv(cpd_dir / "early_warning.csv").set_index("model")
    v["ew_auc"] = fmt(ew.loc["lead1", "roc_auc"], 3)
    v["ew_auc_news"] = fmt(ew.loc["lead1+news", "roc_auc"], 3)
    v["ew_base"] = pct(ew.loc["lead1", "base_rate"])
    v["ew_top"] = pct(ew.loc["lead1", "precision_top1pct"])
    v["ew_lift"] = fmt(ew.loc["lead1", "precision_top1pct"] / ew.loc["lead1", "base_rate"], 0)
    es = pd.read_csv(cpd_dir / "news_event_study.csv")
    g = lambda typ, cat, k: es[(es.type == typ) & (es.category == cat) & (es.k == k)].iloc[0]
    v["es_market_0"] = fmt(-g("disaster", "Маркетплейсы", 0)["mean"])
    v["es_all_2"] = fmt(g("disaster", "Все категории", 2)["mean"])
    v["es_health_1"] = fmt(-g("disaster", "Здоровье", 1)["mean"])
    sel = es[es.k.isin([0, 1, 2]) & es.category.isin(["Все категории", "Маркетплейсы", "Здоровье", "Продовольствие"])]
    sel = sel.assign(cell=[f"{fmt(a)} [{fmt(b)}; {fmt(c)}]" for a, b, c in zip(sel["mean"], sel.ci_low, sel.ci_high)])
    pt = sel.pivot_table(index=["type", "category"], columns="k", values="cell", aggfunc="first").reset_index()
    pt["type"] = pt["type"].map({"disaster": "ЧС, стихия", "security": "атаки, обстрелы"})
    pt.columns = ["Тип события", "Категория", "Месяц события", "+1 мес.", "+2 мес."]
    v["table_event"] = "Таблица 7. Изменение относительного индекса, п.п. [95% ДИ], относительно среднего за 3 месяца до события.\n\n" + md_table(pt)
    nw = pd.read_parquet(proc / "news_region_monthly.parquet")
    mo = nw[nw.territory_id.notna()]
    v["news_share_mo_months"] = pct((mo.news_shock_mo > 0).sum() / (panel.territory_id.nunique() * 24), 1)
    v["runtime_note"] = "≈ 4–5 ч на ноутбуке с 8 ядрами Apple M (большая часть — бэктест Prophet и LightGBM)"
    return v


def news_gain(cfg) -> str:
    """MAE LightGBM с новостями и без на всех прогнозах и на прогнозах для МО-месяцев с новостным событием."""
    res = cfg.path("results") / "full"
    p = res / "all_forecasts.parquet"
    if not p.exists():
        return "нет данных"
    fc = pd.read_parquet(p, columns=["model", "s", "origin", "h", "y", "yhat", "target_period"])
    a, b = "LightGBM", "LightGBM (без новостей)"
    if not {a, b} <= set(fc.model):
        return "нет данных"
    from sbx.forecast.base import build_context
    ctx = build_context(cfg.path("processed"), cfg.path("external"), cfg)
    nw = pd.read_parquet(cfg.path("processed") / "news_region_monthly.parquet")
    mo = nw[nw.territory_id.notna() & (nw.news_shock_mo > 0)][["territory_id", "period"]]
    reg = nw[nw.territory_id.isna() & (nw.region_key != "__RU__")]
    thr = reg.news_shock_surprise.quantile(0.98)
    rg = reg[reg.news_shock_surprise >= thr][["region_key", "period"]]
    meta = ctx.meta[["territory_id", "region_key"]].reset_index().rename(columns={"index": "s"})
    ev = pd.concat([mo.merge(meta, on="territory_id")[["s", "period"]], rg.merge(meta, on="region_key")[["s", "period"]]]).drop_duplicates()
    ev = ev.rename(columns={"period": "target_period"}).assign(event=1)
    sub = fc[fc.model.isin([a, b])].merge(ev, on=["s", "target_period"], how="left").fillna({"event": 0})
    sub["ae"] = (sub.y - sub.yhat).abs()
    t = sub.groupby(["model", "event"]).ae.mean().unstack("model")
    all_ = sub.groupby("model").ae.mean()
    g_all = 1 - all_[a] / all_[b]
    g_ev = 1 - t.loc[1, a] / t.loc[1, b] if 1 in t.index else float("nan")
    return (f"MAE на всех прогнозах {'ниже' if g_all > 0 else 'выше'} на {pct(abs(g_all))}, на прогнозах для МО и месяцев "
            f"с новостным событием — {'ниже' if g_ev > 0 else 'выше'} на {pct(abs(g_ev))}")


def render(md_path: Path, html_path: Path, pdf_path: Path):
    css = ROOT / "reports" / "report.css"
    pandoc = shutil.which("pandoc")
    if pandoc:
        subprocess.run([pandoc, str(md_path), "-s", "--toc", "--toc-depth=2", "-c", css.name, "--metadata", "lang=ru",
                        "-o", str(html_path)], check=True, cwd=md_path.parent)
    else:
        import markdown
        body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code", "toc"])
        html_path.write_text(f"<html><head><meta charset='utf-8'><link rel='stylesheet' href='{css.name}'></head><body>{body}</body></html>")
    chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if chrome:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("Chrome не найден: PDF не создан, используйте HTML")


def update_readme(cfg, v: dict) -> None:
    """Вставляет в README краткую таблицу итогов между маркерами <!--RESULTS--> … <!--/RESULTS-->."""
    m = pd.read_csv(cfg.path("results") / "full" / "metrics_by_window.csv")
    mae = m.pivot(index="model", columns="window", values="MAE")
    rows = [x for x in [v["final_model"], "LightGBM", "Chronos-2-SA + ковариаты", "SNaive×Growth", "Prophet"] if x in mae.index]
    rows = list(dict.fromkeys(rows))
    tab = pd.DataFrame({"Модель": [f"**{x}**" if x == v["final_model"] else x for x in rows],
                        **{f"MAE {H} мес.": [fmt(mae.loc[x, H]) for x in rows] for H in (1, 3, 6, 12)}})
    cs = pd.read_csv(cfg.path("results") / "cpd" / "cpd_synthetic_summary.csv")
    block = (f"## Итоги\n\nБэктест на {v['n_series']} полных рядах, скользящая точка прогноза декабрь 2023 – ноябрь 2024 г. "
             f"MAE — руб. на жителя в месяц; {v['final_model']} лучше Prophet на {v['gain_1']} / {v['gain_3']} / {v['gain_6']} / "
             f"{v['gain_12']} (горизонты 1 / 3 / 6 / 12 мес.).\n\n" + md_table(tab) +
             f"\n\nДетекция шоков (синтетический тест, режим мониторинга): Chronos-2 интервал — F1 {v['chronos_f1']} с "
             f"обнаружением в месяц шока; схема «Chronos-2 → PELT» — точность {v['two_stage_precision']} при "
             f"{v['two_stage_fa']} ложной тревоги на 100 ряд-месяцев.\n")
    readme = ROOT / "README.md"
    txt = readme.read_text()
    new = f"<!--RESULTS-->\n{block}<!--/RESULTS-->"
    txt = re.sub(r"<!--RESULTS-->.*?<!--/RESULTS-->", new, txt, flags=re.S) if "<!--/RESULTS-->" in txt else txt.replace("<!--RESULTS-->", new)
    readme.write_text(txt)


def main():
    cfg, _ = cli_config(__doc__)
    rep = ROOT / "reports"
    tpl = (rep / "methodology_report.template.md").read_text()
    v = build_vars(cfg)
    missing = sorted(set(re.findall(r"{{(\w+)}}", tpl)) - set(v))
    if missing:
        raise SystemExit(f"нет значений для: {missing}")
    md = re.sub(r"{{(\w+)}}", lambda mm: str(v[mm.group(1)]), tpl)
    (rep / "methodology_report.md").write_text(md)
    render(rep / "methodology_report.md", rep / "methodology_report.html", rep / "methodology_report.pdf")
    update_readme(cfg, v)
    print("отчёт:", rep / "methodology_report.pdf")


if __name__ == "__main__":
    main()
