"""Точки изменений на реальных данных, связь с новостями и раннее предупреждение о шоках."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import cpd

OFFLINE = ["pelt", "binseg", "bottomup", "window", "kernel"]
CASES = {  # МО для разбора кейсов: паводки весны 2024 г. и др.
    "Орск (паводок, апрель 2024)": "городской округ город Орск",
    "Новотроицк (паводок, апрель 2024)": "городской округ город Новотроицк",
    "Оренбург (паводок, апрель 2024)": "городской округ город Оренбург",
    "Курган (паводок, апрель 2024)": "городской округ город Курган",
    "Ишим (паводок, апрель 2024)": "городской округ город Ишим",
}


def _offline_all(x, best):
    return {m: cpd.offline_detect(x, m, **best[m]) for m in OFFLINE}


def consensus_breaks(X: np.ndarray, best: dict, min_votes: int = 3, n_jobs: int = 8) -> pd.DataFrame:
    res = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_offline_all)(X[i], best) for i in range(len(X)))
    rows = []
    for i, d in enumerate(res):
        cand = sorted({c for v in d.values() for c in v})
        used = set()
        for c in cand:
            if c in used:
                continue
            votes = sum(any(abs(c - k) <= 1 for k in v) for v in d.values())
            if votes >= min_votes:
                x = X[i]
                before, after = x[max(0, c - 4): c], x[c: c + 4]
                rows.append({"row": i, "cp": c, "votes": votes, "magnitude": float(np.mean(after) - np.mean(before))})
                used |= {c - 1, c, c + 1}
    return pd.DataFrame(rows)


def news_matrix(news: pd.DataFrame, region_keys: np.ndarray, periods: list[str], col: str) -> np.ndarray:
    reg = news[news.territory_id.isna() & (news.region_key != "__RU__")]
    p = reg.pivot_table(index="region_key", columns="period", values=col).reindex(columns=periods)
    return p.reindex(region_keys).fillna(0).to_numpy(float)


def bocpd_news_eval(X, labels: pd.DataFrame, news_surp: np.ndarray, base: dict, betas=(0.0, 0.5, 1.0, 2.0)):
    """Сравнение BOCPD и BOCPD с новостным hazard: h_t = h0·(1 + β·событие_t), событие ∈ {0, 1}."""
    true = labels.groupby("row").cp.apply(list).to_dict()
    out = []
    for beta in betas:
        tp = fp = fn = 0
        delays = []
        for i in range(len(X)):
            hz = np.clip(base["hazard"] * (1 + beta * news_surp[i]), 1e-4, 0.5)
            al = cpd.bocpd_alarm_times(X[i], hz, base["threshold"])
            pred = [c for c, _ in al]
            tr = true.get(i, [])
            a, b, c = cpd.match_cps(pred, tr, 1)
            tp, fp, fn = tp + a, fp + b, fn + c
            for t in tr:
                hits = [tt for cc, tt in al if abs(cc - t) <= 1]
                if hits:
                    delays.append(min(hits) - t)
        s = cpd.f1_scores(tp, fp, fn)
        out.append({"beta": beta, **s, "mean_delay": float(np.mean(delays)) if delays else np.nan,
                    "detected_within_1m": float(np.mean(np.array(delays) <= 1)) if delays else 0.0})
    return pd.DataFrame(out)


def early_warning(X, meta, periods, news, neighbors, out: Path, seed: int = 42):
    """Классификатор вероятности шока в месяце t по информации до t−1 (lead=1) или до t (nowcast)."""
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    S, T = X.shape
    sig = np.array([[cpd.robust_sigma(X[i, :t]) if t >= 4 else np.nan for t in range(T)] for i in range(S)])
    dev = np.full((S, T), np.nan)
    for t in range(3, T):
        dev[:, t] = X[:, t] - np.median(X[:, t - 3: t], axis=1)
    thr = np.maximum(0.10, 4 * sig)
    label = (np.abs(dev) > thr).astype(float)
    label[:, :4] = np.nan
    absdev = np.abs(dev)
    reg = meta.region_key.to_numpy()
    # доля шоков в субъекте и у соседей в предыдущем месяце
    reg_share = pd.DataFrame(np.nan_to_num(label)).groupby(reg).transform("mean").to_numpy()
    nb = neighbors.merge(meta[["territory_id", "category"]].reset_index().rename(columns={"index": "row"}), on="territory_id")
    nb = nb.merge(meta[["territory_id", "category"]].reset_index().rename(columns={"index": "nrow", "territory_id": "neighbor_id"}),
                  on=["neighbor_id", "category"])
    nb_share = np.zeros((S, T))
    cnt = np.zeros(S)
    np.add.at(nb_share, nb.row.to_numpy(), np.nan_to_num(label)[nb.nrow.to_numpy()])
    np.add.at(cnt, nb.row.to_numpy(), 1)
    nb_share = nb_share / np.maximum(cnt, 1)[:, None]
    news_cols = ["news_shock", "news_shock_surprise", "news_security", "news_disaster", "news_utility",
                 "news_econ_neg", "news_tone", "news_n_surprise"]
    NM = {c: news_matrix(news, reg, periods, c) for c in news_cols if c in news}
    mo = news[news.territory_id.notna()]
    tid = meta.territory_id.to_numpy()
    if "news_shock_mo" in mo:
        NM["news_shock_mo"] = mo.pivot_table(index="territory_id", columns="period", values="news_shock_mo") \
            .reindex(index=tid, columns=periods).fillna(0).to_numpy(float)
    cat = meta.category.astype("category").cat.codes.to_numpy()

    def feats(t_feat: np.ndarray, t_news: np.ndarray, rows: np.ndarray, use_news: bool) -> pd.DataFrame:
        f = {"cat": cat[rows], "month": np.array([int(p[5:7]) for p in periods])[t_feat + 1],
             "vol": sig[rows, t_feat], "last_absdev": absdev[rows, t_feat],
             "max_absdev_3m": np.nanmax(np.stack([absdev[rows, np.maximum(t_feat - k, 0)] for k in range(3)]), axis=0),
             "level_rel": X[rows, t_feat], "reg_share": reg_share[rows, t_feat], "nb_share": nb_share[rows, t_feat],
             "log_population": meta.log_population.to_numpy()[rows], "market_access": meta.market_access.to_numpy()[rows],
             "monotown": meta.monotown_status.to_numpy()[rows]}
        if use_news:
            for c, M in NM.items():
                f[c] = M[rows, t_news]
                f[c + "_l1"] = M[rows, np.maximum(t_news - 1, 0)]
        return pd.DataFrame(f)

    rows_all = np.repeat(np.arange(S), T)
    t_all = np.tile(np.arange(T), S)
    ok = ~np.isnan(label[rows_all, t_all]) & (t_all >= 5)
    rows_all, t_all = rows_all[ok], t_all[ok]
    y = label[rows_all, t_all]
    split = periods.index("2024-03")
    tr, te = t_all <= split, t_all > split
    results, preds = [], {}
    for mode, news_lag in (("lead1", 1), ("nowcast", 0)):
        for use_news in (False, True):
            if mode == "nowcast" and not use_news:
                continue   # без новостей nowcast совпадает с lead1
            X_ = feats(t_all - 1, t_all - news_lag, rows_all, use_news)
            params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed)
            bst = lgb.train(params, lgb.Dataset(X_[tr], y[tr], categorical_feature=["cat"]), num_boost_round=400)
            p = bst.predict(X_[te])
            k = max(1, int(0.01 * te.sum()))
            top = np.argsort(-p)[:k]
            name = f"{mode}{'+news' if use_news else ''}"
            preds[name] = p
            imp = pd.Series(bst.feature_importance("gain"), index=X_.columns).sort_values(ascending=False)
            results.append({"model": name, "roc_auc": roc_auc_score(y[te], p), "pr_auc": average_precision_score(y[te], p),
                            "base_rate": float(y[te].mean()), "precision_top1pct": float(y[te][top].mean()),
                            "top_features": ", ".join(imp.index[:6])})
    res = pd.DataFrame(results)
    res.to_csv(out / "early_warning.csv", index=False)
    lab = pd.DataFrame({"row": rows_all[te], "t": t_all[te], "y": y[te], **{k: v for k, v in preds.items()}})
    lab.to_parquet(out / "early_warning_preds.parquet")
    return res, label


def run_real(cfg, W: pd.DataFrame, X: np.ndarray, best: dict, out: Path):
    proc = cfg.path("processed")
    periods = list(W.columns)
    meta = W.index.to_frame(index=False).merge(pd.read_parquet(proc / "static.parquet"), on="territory_id", how="left")
    news_path = proc / "news_region_monthly.parquet"
    news = pd.read_parquet(news_path) if news_path.exists() else None
    # 1) консенсусные точки изменений (≥3 из 5 офлайн-методов)
    cb = consensus_breaks(X, best, n_jobs=cfg.project["n_jobs"])
    cb["territory_id"] = meta.territory_id.to_numpy()[cb.row]
    cb["category"] = meta.category.to_numpy()[cb.row]
    cb["region"] = meta.region_name.to_numpy()[cb.row]
    cb["mo_name"] = meta.mo_name.to_numpy()[cb.row]
    cb["period"] = np.array(periods)[cb.cp]
    cb.to_csv(out / "real_consensus_breaks.csv", index=False)
    big = cb[cb.magnitude.abs() >= 0.10]
    print(f"консенсусных точек: {len(cb):,}; из них |сдвиг| ≥ 10%: {len(big):,}")
    summary = {"n_series": int(len(X)), "n_breaks": int(len(cb)), "n_big": int(len(big)),
               "by_period": big.groupby("period").size().to_dict(),
               "by_category": big.groupby("category").size().to_dict()}
    if news is not None:
        surp = news_matrix(news, meta.region_key.to_numpy(), periods, "news_shock_surprise")
        mo = news[news.territory_id.notna()]
        mo_shock = mo.pivot_table(index="territory_id", columns="period", values="news_shock_mo") \
            .reindex(index=meta.territory_id.to_numpy(), columns=periods).fillna(0).to_numpy(float)
        # новостное событие: всплеск шоковых новостей в субъекте (верхние 2% субъект-месяцев)
        # или шоковая новость с упоминанием самого МО
        thr = np.nanquantile(surp[:, 2:], 0.98)
        event = (surp >= thr) | (mo_shock > 0)
        brk = np.zeros_like(X, dtype=bool)
        brk[big.row.to_numpy(), big.cp.to_numpy()] = True
        brk_next = brk.copy()
        brk_next[:, :-1] |= brk[:, 1:]                       # сдвиг в месяце события или следующем
        ev, br = event[:, 2:-1].ravel(), brk_next[:, 2:-1].ravel()
        p1, p0 = br[ev].mean(), br[~ev].mean()
        rng = np.random.default_rng(0)
        boot = []
        i1, i0 = np.where(ev)[0], np.where(~ev)[0]
        for _ in range(500):
            b1 = br[rng.choice(i1, len(i1))].mean()
            b0 = br[rng.choice(i0, len(i0))].mean()
            boot.append(b1 / b0)
        summary.update({"news_event_threshold": float(thr), "n_event_mo_months": int(ev.sum()),
                        "p_break_given_event": float(p1), "p_break_given_no_event": float(p0),
                        "lift": float(p1 / p0), "lift_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]})
        print(f"P(сдвиг | новостное событие) = {p1:.3f}, P(сдвиг | нет события) = {p0:.3f}, lift = {p1 / p0:.2f} "
              f"[{np.quantile(boot, 0.025):.2f}; {np.quantile(boot, 0.975):.2f}]")
        # 2) BOCPD с новостным hazard против обычного (псевдоразметка — консенсус офлайн-методов, без новостей)
        bn = bocpd_news_eval(X, big, event.astype(float), best["bocpd"], betas=(0.0, 1.0, 3.0, 6.0))
        bn.to_csv(out / "real_bocpd_news.csv", index=False)
        print(bn.round(3).to_string())
        # 2b) event study по новостям с прямым упоминанием МО
        tagged = pd.read_parquet(proc / "news_tagged.parquet")
        es = news_event_study(X, meta, periods, tagged)
        es.to_csv(out / "news_event_study.csv", index=False)
        print(es[es.k.isin([0, 1])].round(2).to_string())
        # 3) раннее предупреждение
        ew, label = early_warning(X, meta, periods, news, pd.read_parquet(proc / "neighbors.parquet"), out, cfg.project["seed"])
        print(ew.round(3).to_string())
    # 4) кейсы
    cases = []
    for title, mo in CASES.items():
        rows = np.where((meta.mo_name == mo).to_numpy())[0]
        for r in rows:
            x = X[r]
            item = {"case": title, "mo": mo, "category": meta.category.iat[r], "region": meta.region_name.iat[r],
                    "periods": periods, "x": x.tolist(), "y": W.to_numpy()[r].tolist(),
                    "offline": {m: cpd.offline_detect(x, m, **best[m]) for m in OFFLINE},
                    "bocpd": cpd.bocpd_alarm_times(x, best["bocpd"]["hazard"], best["bocpd"]["threshold"]),
                    "cusum": cpd.cusum(x, **best["cusum"])}
            if news is not None:
                rk = meta.region_key.iat[r]
                item["news_shock"] = news_matrix(news, np.array([rk]), periods, "news_shock")[0].tolist()
                item["news_surprise"] = news_matrix(news, np.array([rk]), periods, "news_shock_surprise")[0].tolist()
                thr_case = summary.get("news_event_threshold", 1.5)
                hz = np.clip(best["bocpd"]["hazard"] * (1 + 3.0 * (np.array(item["news_surprise"]) >= thr_case)), 1e-4, 0.5)
                item["bocpd_news"] = cpd.bocpd_alarm_times(x, hz, best["bocpd"]["threshold"])
            cases.append(item)
    (out / "real_cases.json").write_text(json.dumps(cases, ensure_ascii=False, default=float))
    (out / "real_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


def news_event_study(X: np.ndarray, meta: pd.DataFrame, periods: list[str], tagged: pd.DataFrame,
                     types=("ev_disaster", "ev_security"), min_headlines: int = 3, window=(-2, 3)) -> pd.DataFrame:
    """Event study: как относительный индекс МО меняется вокруг месяца, когда МО упомянуто
    в ≥ min_headlines шоковых новостях данного типа. Отклонение считается от среднего за 3
    месяца до события; доверительный интервал — ±1.96·SE по событиям."""
    t = tagged[tagged.mos.str.len() > 0].copy()
    t["period"] = t.date.str[:7]
    e = t.explode("mos")
    tid = meta.territory_id.to_numpy()
    rows = []
    for typ in types:
        cnt = e[e[typ] == 1].groupby(["mos", "period"]).size()
        ev = cnt[cnt >= min_headlines].reset_index()
        ev = ev[ev.period.between(periods[3], periods[-1 - window[1]])]
        for _, r in ev.iterrows():
            k0 = periods.index(r.period)
            for i in np.where(tid == r.mos)[0]:
                base = np.mean(X[i, k0 - 3: k0])
                for k in range(window[0], window[1] + 1):
                    rows.append({"type": typ[3:], "category": meta.category.iat[i], "k": k,
                                 "dev": (X[i, k0 + k] - base) * 100, "event": f"{r.mos}:{r.period}"})
    d = pd.DataFrame(rows)
    g = d.groupby(["type", "category", "k"]).dev.agg(["mean", "sem", "count"]).reset_index()
    g["ci_low"], g["ci_high"] = g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"]
    g["n_events"] = g.groupby(["type"])["count"].transform("max")
    return g
