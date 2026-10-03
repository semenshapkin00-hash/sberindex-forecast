"""Шаг 6. Графики для отчёта (reports/figures/*.svg|png) и данные для лендинга (docs/data.json).

    python scripts/06_make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import viz  # noqa: E402
from sbx.config import ROOT, cli_config  # noqa: E402

plt = viz.plt
FINAL = "Ансамбль"          # итоговая модель (обновляется по результатам в main)
BASE = "Prophet"
SHOW = ["Ансамбль", "LightGBM (только ряд)", "LightGBM", "LightGBM (без новостей)", "LightGBM + Chronos-2 (стэкинг)", "SNaive×Growth",
        "NatSeasonal", "Chronos-2-SA + ковариаты", "Theta-SA", "Prophet", "Naive"]


def fig_mae_windows(m: pd.DataFrame, out: Path, final: str):
    piv = m.pivot(index="model", columns="window", values="MAE")
    piv = piv.loc[[x for x in SHOW if x in piv.index]]
    order = piv.mean(axis=1).sort_values(ascending=False).index   # сверху — лучшие
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.9), sharey=True)
    for ax, H in zip(axes, [1, 3, 6, 12]):
        vals = piv.loc[order, H]
        colors = [viz.SERIES[0] if i == final else viz.SERIES[1] if i == BASE else viz.OTHER for i in order]
        ax.barh(range(len(order)), vals, color=colors, height=0.66, edgecolor=viz.SURFACE, linewidth=1)
        for k, v in enumerate(vals):
            ax.text(v + vals.max() * 0.015, k, f"{v:.0f}", va="center", fontsize=8.5, color=viz.INK2)
        ax.set_title(f"горизонт {H} мес.")
        ax.set_xlim(0, vals.max() * 1.22)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MAE, руб./жителя")
    axes[0].set_yticks(range(len(order)), order)
    fig.suptitle("Ошибка прогноза (MAE) по окнам прогнозирования: синий — итоговая модель, оранжевый — Prophet",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    viz.save(fig, out / "mae_by_window.svg")


def fig_mae_steps(s: pd.DataFrame, out: Path, final: str):
    models = [final, "LightGBM", "SNaive×Growth", "Chronos-2-SA + ковариаты", BASE]
    models = [x for x in dict.fromkeys(models) if x in set(s.model)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, mname in enumerate(models):
        d = s[s.model == mname].sort_values("h")
        color = viz.SERIES[1] if mname == BASE else viz.SERIES[[0, 2, 3, 4, 6][min(k, 4)]]
        ax.plot(d.h, d.MAE, color=color, marker="o", markersize=4.5, label=mname)
        ax.text(d.h.iloc[-1] + 0.15, d.MAE.iloc[-1], mname, color=viz.INK2, fontsize=8.5, va="center")
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.7, 15.5)
    ax.set_xlabel("шаг прогноза h, месяцев")
    ax.set_ylabel("MAE, руб./жителя")
    ax.set_title("Ошибка в зависимости от дальности прогноза (все точки прогноза 2023-12 … 2024-11)")
    ax.legend(loc="upper left", fontsize=8.5)
    viz.save(fig, out / "mae_by_step.svg")


def fig_gain_category(mc: pd.DataFrame, out: Path, final: str):
    a = mc[mc.model == final].set_index(["category", "window"]).MAE
    b = mc[mc.model == BASE].set_index(["category", "window"]).MAE
    g = (1 - a / b).unstack("window") * 100
    cats = ["Все категории", "Продовольствие", "Маркетплейсы", "Общественное питание", "Транспорт", "Здоровье"]
    g = g.loc[[c for c in cats if c in g.index]]
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", viz.DIVERGING)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    lim = max(10, np.nanmax(np.abs(g.to_numpy())))
    ax.imshow(g.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.iat[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > lim * 0.55 else viz.INK)
    ax.set_xticks(range(g.shape[1]), [f"{c} мес." for c in g.columns])
    ax.set_yticks(range(g.shape[0]), g.index)
    ax.grid(False)
    ax.set_title(f"Снижение MAE относительно Prophet: {final}")
    viz.save(fig, out / "gain_by_category.svg")


def fig_cpd_tradeoff(cs: pd.DataFrame, out: Path):
    d = cs[cs["mode"] == "мониторинг"].copy()
    d = d[d.recall > 0]
    d["label"] = d.method + np.where(d.method.str.contains("интервал"), " " + d.params.astype(str), "")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fa = d.false_alarms_per_100_months.clip(lower=0.05)
    ax.scatter(fa, d.F1, s=60, color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
    for x, y, lab, dl in zip(fa, d.F1, d.label, d.mean_delay):
        ax.annotate(f"{lab}\nзадержка {dl:.1f} мес.", (x, y), textcoords="offset points", xytext=(7, -2),
                    fontsize=7.6, color=viz.INK2)
    ax.set_xscale("log")
    ax.set_xlabel("ложные тревоги на 100 ряд-месяцев (ряды без шока), лог. шкала")
    ax.set_ylabel("F1 (допуск ±1 мес.)")
    ax.set_title("Детекторы в режиме ежемесячного мониторинга: точность обнаружения vs ложные тревоги")
    viz.save(fig, out / "cpd_tradeoff.svg")


def fig_cpd_by_type(bt: pd.DataFrame, out: Path):
    kinds = {"level": "сдвиг уровня", "slope": "постепенный сдвиг (4 мес.)", "transient": "временный шок (3 мес.)"}
    methods = ["PELT", "KernelCPD (RBF)", "BOCPD", "CUSUM"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (k, title) in zip(axes, kinds.items()):
        for i, mname in enumerate(methods):
            d = bt[(bt.kind == k) & (bt.method == mname)].sort_values("mag")
            ax.plot(d.mag * 100, d.recall, marker="o", markersize=4.5, color=viz.SERIES[i], label=mname)
        ax.set_title(title)
        ax.set_xlabel("величина шока, %")
        ax.set_xticks([10, 20, 35])
    axes[0].set_ylabel("полнота обнаружения")
    axes[-1].legend(loc="lower right", fontsize=8)
    viz.save(fig, out / "cpd_by_type.svg")


def _dedupe(cps, gap=1):
    out = []
    for c in sorted(cps):
        if not out or c - out[-1] > gap:
            out.append(c)
    return out


def fig_cases(cases: list, out: Path):
    """Орск (паводок апреля 2024 г.): категории трат + новостной фон; маркетплейсы в городах паводка."""
    orsk = {c["category"]: c for c in cases if c["mo"] == "городской округ город Орск"}
    if not orsk:
        return
    cats = ["Все категории", "Маркетплейсы", "Здоровье", "Общественное питание"]
    per = orsk[cats[0]]["periods"]
    x = np.arange(len(per))
    ev = per.index("2024-04")
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.2], hspace=0.35)
    for j, cat in enumerate(cats):
        c = orsk[cat]
        ax = fig.add_subplot(gs[0, j])
        ax.axvspan(ev - 0.5, ev + 0.5, color=viz.BLUE_RAMP[0], zorder=0)
        ax.plot(x, np.array(c["x"]) * 100, color=viz.SERIES[0], marker="o", markersize=3)
        votes = [k for k in sorted({k for v in c["offline"].values() for k in v})
                 if sum(any(abs(k - q) <= 1 for q in v) for v in c["offline"].values()) >= 3]
        for cp in _dedupe(votes):
            ax.axvline(cp - 0.5, color=viz.SERIES[1], linewidth=1.3, linestyle="--")
        for cp, tal in c.get("bocpd", []):
            ax.plot(tal, c["x"][tal] * 100, marker="v", color=viz.SERIES[7], markersize=7, linestyle="none")
        ax.set_title(cat, fontsize=10)
        ax.set_xticks(x[::6], [p[2:] for p in per[::6]], fontsize=8)
        if j == 0:
            ax.set_ylabel("отклонение от сопоставимых МО, %")
    axn = fig.add_subplot(gs[1, :])
    axn.bar(x, orsk[cats[0]].get("news_shock", [0] * len(x)), color=viz.BLUE_RAMP[3], width=0.7)
    axn.set_xticks(x, [p[2:] for p in per], fontsize=7.5)
    axn.set_ylabel("шоковые новости\nсубъекта, log(1+n)")
    fig.suptitle("Орск, паводок апреля 2024 г. (выделен): шок проявляется в структуре трат. Пунктир — точка изменения "
                 "(≥3 из 5 офлайн-методов), ▼ — тревога BOCPD", x=0.01, ha="left", fontsize=10)
    viz.save(fig, out / "case_orsk.svg")

    # маркетплейсы в городах, пострадавших от паводка
    cities = [c for c in cases if c["category"] == "Маркетплейсы"]
    if cities:
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ax.axvspan(ev - 0.5, ev + 0.5, color=viz.BLUE_RAMP[0], zorder=0)
        for i, c in enumerate(cities):
            xx = np.array(c["x"]) * 100
            xx = xx - np.mean(xx[ev - 3: ev])
            name = c["case"].split(" (")[0]
            ax.plot(x[ev - 6: ev + 6], xx[ev - 6: ev + 6], color=viz.SERIES[i], marker="o", markersize=3.5, label=name)
        ax.axhline(0, color=viz.AXIS, linewidth=1)
        ax.set_xticks(x[ev - 6: ev + 6], [p[2:] for p in per[ev - 6: ev + 6]], fontsize=8)
        ax.set_ylabel("п.п. к среднему за 3 мес. до паводка")
        ax.set_title("«Маркетплейсы» в городах паводка 2024 г.: провал в апреле и отскок в мае")
        ax.legend(fontsize=8, ncol=3, loc="lower left")
        viz.save(fig, out / "case_marketplaces.svg")


def fig_april2023(proc: Path, out: Path):
    p = pd.read_parquet(proc / "panel.parquet")
    st = pd.read_parquet(proc / "static.parquet")
    p = p.merge(st[["territory_id", "region_name", "mo_type"]], on="territory_id")
    fed = p.region_name.isin(["Москва", "Санкт-Петербург"])
    p["grp"] = np.where(fed, "города фед. значения", np.where(p.mo_type == 0, "городские округа", "районы и мун. округа"))
    cats = ["Здоровье", "Общественное питание", "Транспорт"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, cat in zip(axes, cats):
        d = p[(p.category == cat) & (p.period <= "2023-12")].pivot_table(index="grp", columns="period", values="value", aggfunc="median")
        d = np.log(d.div(d["2023-03"], axis=0)) * 100
        for i, g in enumerate(["города фед. значения", "городские округа", "районы и мун. округа"]):
            ax.plot(range(d.shape[1]), d.loc[g], color=viz.SERIES[i], marker="o", markersize=3.5, label=g)
        ax.axvline(3, color=viz.MUTED, linestyle=":", linewidth=1)
        ax.set_title(cat)
        ax.set_xticks(range(0, d.shape[1], 2), [c[5:] + "." + c[2:4] for c in d.columns[::2]], fontsize=8)
    axes[0].set_ylabel("лог-изменение к марту 2023, %")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Апрель 2023: синхронное расхождение «город — село» (медианы по группам МО)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    viz.save(fig, out / "april2023.svg")


def fig_news_heatmap(proc: Path, out: Path):
    n = pd.read_parquet(proc / "news_region_monthly.parquet")
    st = pd.read_parquet(proc / "static.parquet")
    names = st.drop_duplicates("region_key").set_index("region_key").region_name
    r = n[n.territory_id.isna() & (n.region_key != "__RU__") & (n.period >= "2023-01")]
    top = r.groupby("region_key").news_shock_surprise.max().sort_values(ascending=False).head(14).index
    piv = r[r.region_key.isin(top)].pivot_table(index="region_key", columns="period", values="news_shock_surprise").loc[top]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb"] + viz.BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.imshow(piv.clip(lower=0).to_numpy(), cmap=cmap, aspect="auto")
    ax.set_yticks(range(len(top)), [names.get(k, k) for k in top], fontsize=8.5)
    ax.set_xticks(range(0, piv.shape[1], 2), [c[2:] for c in piv.columns[::2]], fontsize=8)
    ax.grid(False)
    ax.set_title("«Новостной сюрприз» по шоковым событиям: субъекты с самыми сильными всплесками (Интерфакс + Lenta.ru)")
    viz.save(fig, out / "news_heatmap.svg")


def fig_importance(res: Path, out: Path):
    p = res / "importance_lightgbm.csv"
    if not p.exists():
        return
    imp = pd.read_csv(p, index_col=0).mean(axis=1)
    def grp(f):
        if f.startswith("news"):
            return "новости"
        if f.startswith("cp_"):
            return "детектор изменений (as-of)"
        if f.startswith("w_"):
            return "погода"
        if f.startswith("nb_") or f == "own_vs_nb_mom3":
            return "соседи по дорогам"
        if f.endswith("_target") or f == "workdays_diff":
            return "календарь"
        if f in ("fm", "fm_width"):
            return "Chronos-2 (стэкинг)"
        if f.startswith(("nat_", "cs_", "reg_", "cpi", "wage", "arrears", "key_rate", "own_vs_reg")) or f in (
                "log_population", "urban_share", "market_access", "is_capital", "mo_type", "monotown_status", "lat", "lon", "region"):
            return "макро, регион, характеристики МО"
        return "собственная история ряда"
    g = imp.groupby(imp.index.map(grp)).sum()
    g = (g / g.sum() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(g.index, g.values, color=viz.SERIES[0], height=0.62)
    for k, v in enumerate(g.values):
        ax.text(v + 0.6, k, f"{v:.1f}%", va="center", fontsize=8.5, color=viz.INK2)
    ax.set_xlabel("доля суммарного прироста качества (gain), %")
    ax.set_title("Вклад групп признаков в LightGBM (среднее по точкам прогноза)")
    ax.grid(axis="y", visible=False)
    viz.save(fig, out / "importance.svg")


def fig_early_warning(cpd_dir: Path, out: Path):
    p = cpd_dir / "early_warning.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    labels = {"lead1": "за месяц, без новостей", "lead1+news": "за месяц, + новости",
              "nowcast+news": "в текущем месяце, + новости"}
    d["label"] = d.model.map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    for ax, col, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC (доля шоков {:.1%})".format(d.base_rate.iloc[0])]):
        ax.barh(d.label, d[col], color=[viz.OTHER, viz.SERIES[0], viz.SERIES[2]][: len(d)], height=0.6)
        for k, v in enumerate(d[col]):
            ax.text(v + 0.005, k, f"{v:.3f}", va="center", fontsize=8.5, color=viz.INK2)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(1.0 if col == "roc_auc" else d[col].max() * 1.3, 0.1))
    fig.suptitle("Раннее предупреждение о шоке в МО (тест: апрель–декабрь 2024)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    viz.save(fig, out / "early_warning.svg")


def fig_event_study(cpd_dir: Path, out: Path):
    p = cpd_dir / "news_event_study.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    titles = {"disaster": "ЧС и стихийные бедствия", "security": "атаки и обстрелы"}
    cats = ["Все категории", "Маркетплейсы", "Здоровье", "Продовольствие"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (typ, title) in zip(axes, titles.items()):
        for i, cat in enumerate(cats):
            g = d[(d.type == typ) & (d.category == cat)].sort_values("k")
            off = (i - 1.5) * 0.07
            ax.errorbar(g.k + off, g["mean"], yerr=1.96 * g["sem"], color=viz.SERIES[i], marker="o", markersize=4.5,
                        capsize=2.5, linewidth=1.6, elinewidth=1, label=cat)
        ax.axhline(0, color=viz.AXIS, linewidth=1)
        ax.axvline(0, color=viz.MUTED, linestyle=":", linewidth=1)
        n = int(d[d.type == typ].n_events.max())
        ax.set_title(f"{title} (эпизодов: {n})")
        ax.set_xlabel("месяц относительно события")
    axes[0].set_ylabel("изменение отн. индекса, п.п.")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Event study: расходы МО вокруг месяцев с шоковыми новостями о самом МО (95% ДИ)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    viz.save(fig, out / "event_study.svg")


def main():
    cfg, _ = cli_config(__doc__)
    viz.setup()
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    res = cfg.path("results") / "full"
    proc = cfg.path("processed")
    cpd_dir = cfg.path("results") / "cpd"
    m = pd.read_csv(res / "metrics_by_window.csv")
    final = FINAL if FINAL in set(m.model) else m[m.window == 12].sort_values("MAE").model.iloc[0]
    fig_mae_windows(m, out, final)
    fig_mae_steps(pd.read_csv(res / "metrics_by_step.csv"), out, final)
    fig_gain_category(pd.read_csv(res / "metrics_by_window_category.csv"), out, final)
    fig_importance(res, out)
    fig_april2023(proc, out)
    if (proc / "news_region_monthly.parquet").exists():
        fig_news_heatmap(proc, out)
    if (cpd_dir / "cpd_synthetic_summary.csv").exists():
        fig_cpd_tradeoff(pd.read_csv(cpd_dir / "cpd_synthetic_summary.csv"), out)
        fig_cpd_by_type(pd.read_csv(cpd_dir / "cpd_synthetic_by_type.csv"), out)
    if (cpd_dir / "real_cases.json").exists():
        fig_cases(json.loads((cpd_dir / "real_cases.json").read_text()), out)
    fig_early_warning(cpd_dir, out)
    fig_event_study(cpd_dir, out)
    print("графики:", sorted(p.name for p in out.glob("*.svg")))


if __name__ == "__main__":
    main()
