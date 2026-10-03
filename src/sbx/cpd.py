"""Обнаружение точек структурных изменений (change point detection).

Сигнал. Для ряда МО строится «относительный индекс»
    x_t = log y_t − median_j(log y_jt),
где медиана берётся по группе сопоставимых МО (та же категория трат и тот же тип поселения)
в месяце t. Общая сезонность, инфляция, тренд и общегрупповые сдвиги взаимно сокращаются,
и локальный шок МО превращается в сдвиг уровня x_t.

Методы
  офлайн (ретроспективная сегментация, ruptures): PELT, BinSeg, BottomUp, Window, KernelCPD;
  онлайн (последовательные, для раннего обнаружения): CUSUM, Page–Hinkley, BOCPD
  (Adams & MacKay, 2007), BOCPD с новостным hazard (априорная вероятность смены режима
  растёт при новостном шоке в субъекте), детектор выхода факта за прогнозный интервал.

Онлайн-методы выдают момент тревоги (alarm) ≥ момента изменения; задержка = alarm − cp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special


# ------------------------------------------------------------------ подготовка сигнала
def peer_groups(meta: pd.DataFrame) -> np.ndarray:
    """Группа сопоставимых МО: категория × тип поселения (города фед. значения / городские
    округа / районы и муниципальные округа). Нормировка внутри группы убирает общие для
    группы сдвиги (напр. синхронное расхождение «город–село» в апреле 2023 г.)."""
    fed = meta.region_name.isin(["Москва", "Санкт-Петербург", "Севастополь"]).to_numpy()
    kind = np.where(fed, "fed", np.where(meta.mo_type.to_numpy() == 0, "urban", "rural"))
    return (meta.category.astype(str) + "|" + kind).to_numpy()


def relative_index(Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """x_t = log y_t − медиана log y_t по группе сопоставимых МО (см. peer_groups)."""
    L = np.log(Y)
    X = np.empty_like(L)
    for c in np.unique(groups):
        m = groups == c
        X[m] = L[m] - np.nanmedian(L[m], axis=0)
    return X


def robust_sigma(x: np.ndarray) -> float:
    d = np.diff(x[~np.isnan(x)])
    if len(d) < 2:
        return 1.0
    s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    return max(s, 1e-3)


# ------------------------------------------------------------------ офлайн (ruptures)
def offline_detect(x: np.ndarray, method: str, pen_mult: float = 3.0, min_size: int = 2) -> list[int]:
    """Возвращает индексы начала новых режимов (cp ∈ [1, n−1])."""
    import ruptures as rpt

    x = np.asarray(x, float)
    n = len(x)
    sig = robust_sigma(x)
    z = ((x - np.nanmedian(x)) / sig).reshape(-1, 1)
    pen = pen_mult * np.log(n)
    if method == "pelt":
        algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(z)
    elif method == "binseg":
        algo = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(z)
    elif method == "bottomup":
        algo = rpt.BottomUp(model="l2", min_size=min_size, jump=1).fit(z)
    elif method == "window":
        algo = rpt.Window(width=6, model="l2", min_size=min_size, jump=1).fit(z)
    elif method == "kernel":
        algo = rpt.KernelCPD(kernel="rbf", min_size=min_size, jump=1).fit(z)
        pen = pen_mult * 0.15 * np.log(n)   # шкала стоимости ядра другая — масштаб подобран на валидации
    else:
        raise ValueError(method)
    bk = algo.predict(pen=pen)
    return [b for b in bk if 0 < b < n]


# ------------------------------------------------------------------ онлайн
def cusum(x: np.ndarray, k: float = 0.5, h: float = 4.0, warmup: int = 4) -> list[int]:
    """Двусторонний CUSUM по стандартизованным отклонениям от среднего текущего режима."""
    sig = robust_sigma(x[: max(warmup + 2, 6)]) if len(x) > 6 else robust_sigma(x)
    alarms, start = [], 0
    gp = gn = 0.0
    for t in range(len(x)):
        if t - start < warmup:
            continue
        mu = np.mean(x[start: t]) if t > start else x[start]
        z = (x[t] - mu) / sig
        gp, gn = max(0.0, gp + z - k), max(0.0, gn - z - k)
        if gp > h or gn > h:
            alarms.append(t)
            start, gp, gn = t, 0.0, 0.0
    return alarms


def page_hinkley(x: np.ndarray, delta: float = 0.5, lam: float = 5.0, warmup: int = 4) -> list[int]:
    sig = robust_sigma(x)
    alarms, start = [], 0
    while True:
        found = None
        mt_u = mt_d = 0.0
        min_u = max_d = 0.0
        for t in range(start + warmup, len(x)):
            mu = np.mean(x[start: t])
            z = (x[t] - mu) / sig
            mt_u += z - delta
            mt_d += z + delta
            min_u, max_d = min(min_u, mt_u), max(max_d, mt_d)
            if mt_u - min_u > lam or max_d - mt_d > lam:
                found = t
                break
        if found is None:
            return alarms
        alarms.append(found)
        start = found


def bocpd(x: np.ndarray, hazard: np.ndarray | float = 0.05, mu0: float | None = None, kappa0: float = 1.0,
          alpha0: float = 2.0, beta0: float | None = None) -> np.ndarray:
    """BOCPD (Adams & MacKay 2007) с нормальным правдоподобием и сопряжённым NIG-априором.

    hazard — скаляр или массив длины n (переменная во времени априорная вероятность смены
    режима, напр. зависящая от новостей). Возвращает матрицу R (n+1 × n+1) апостериорных
    вероятностей длины текущего режима; P(cp в t) = R[t+1, 0].
    """
    x = np.asarray(x, float)
    n = len(x)
    H = np.full(n, hazard) if np.isscalar(hazard) else np.asarray(hazard, float)
    sig = robust_sigma(x)
    mu0 = np.nanmedian(x[: min(n, 4)]) if mu0 is None else mu0
    beta0 = alpha0 * sig ** 2 if beta0 is None else beta0
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    for t in range(n):
        # предсказательное распределение Стьюдента для каждой длины режима
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        zz = (x[t] - mu) / scale
        logp = (special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(np.pi * df) - np.log(scale)
                - (df + 1) / 2 * np.log1p(zz ** 2 / df))
        pred = np.exp(logp - logp.max())
        growth = R[t, : t + 1] * pred * (1 - H[t])
        cp = np.sum(R[t, : t + 1] * pred * H[t])
        R[t + 1, 1: t + 2] = growth
        R[t + 1, 0] = cp
        R[t + 1] /= R[t + 1].sum()
        # обновление параметров (длина режима 0 — априор)
        mu_new = (kappa * mu + x[t]) / (kappa + 1)
        kappa_new = kappa + 1
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
    return R


def bocpd_alarms(x: np.ndarray, hazard=0.05, threshold: float = 0.5, lag: int = 1, warmup: int = 3) -> list[int]:
    """Тревога в момент t, если P(длина режима ≤ lag | x_1..t) > threshold.

    lag=1 допускает подтверждение одним наблюдением после смены (тревога с задержкой ≤ 1).
    Возвращает оценку момента изменения (t − длина режима), а не момента тревоги;
    момент тревоги — в bocpd_alarm_times.
    """
    return [cp for cp, _ in bocpd_alarm_times(x, hazard, threshold, lag, warmup)]


def bocpd_alarm_times(x, hazard=0.05, threshold=0.5, lag=1, warmup=3) -> list[tuple[int, int]]:
    R = bocpd(x, hazard)
    n = len(x)
    out, last = [], -10
    for t in range(warmup, n):
        post = R[t + 1, : lag + 2]
        p = post[: lag + 1].sum()
        if p > threshold and t - last > lag + 1:
            rl = int(np.argmax(post))
            cp = t - rl + 1 if rl > 0 else t
            out.append((cp, t))
            last = t
    return out


def interval_alarms(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, consecutive: int = 1) -> list[int]:
    """Тревога при выходе факта за прогнозный интервал [lo, hi] (consecutive раз подряд)."""
    out = (y < lo) | (y > hi)
    alarms, run = [], 0
    for t, o in enumerate(out):
        run = run + 1 if o else 0
        if run == consecutive:
            alarms.append(t - consecutive + 1)
    return alarms


# ------------------------------------------------------------------ синтетика и метрики
def inject(x: np.ndarray, kind: str, tau: int, delta: float, rng: np.random.Generator) -> np.ndarray:
    """Вносит в ряд шок типа kind в момент tau. delta — величина в лог-единицах (0.2 ≈ 20%)."""
    y = x.copy()
    sign = rng.choice([-1, 1])
    n = len(y)
    if kind == "level":
        y[tau:] += sign * delta
    elif kind == "slope":
        ramp = np.minimum(np.arange(n - tau) + 1, 4) / 4.0
        y[tau:] += sign * delta * ramp
    elif kind == "transient":
        y[tau: tau + 3] += sign * delta
    return y


def match_cps(pred: list[int], true: list[int], margin: int = 1) -> tuple[int, int, int]:
    """TP/FP/FN с допуском ±margin (каждая истинная точка сопоставляется не более одного раза)."""
    used, tp = set(), 0
    for p in sorted(pred):
        cand = [t for t in true if abs(t - p) <= margin and t not in used]
        if cand:
            used.add(min(cand, key=lambda t: abs(t - p)))
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def f1_scores(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def covering(pred: list[int], true: list[int], n: int) -> float:
    """Метрика Covering (van den Burg & Williams, 2020) — сходство разбиений на сегменты."""
    def segs(cps):
        b = [0] + sorted(set(c for c in cps if 0 < c < n)) + [n]
        return [set(range(b[i], b[i + 1])) for i in range(len(b) - 1)]
    G, P = segs(true), segs(pred)
    tot = 0.0
    for A in G:
        tot += len(A) * max(len(A & B) / len(A | B) for B in P)
    return tot / n


# ------------------------------------------------------------------ признаки для прогноза (as-of)
def _asof_one(x: np.ndarray, pen_mult: float, hazard: float, start: int = 5) -> np.ndarray:
    """Для каждого t — что знает монитор к концу месяца t (только x[:t+1]):
    давность последнего найденного сдвига (PELT), его величина, P(смена режима ≤ 1 мес.) по BOCPD."""
    n = len(x)
    out = np.full((n, 3), np.nan)
    R = bocpd(x, hazard)
    for t in range(n):
        out[t, 2] = R[t + 1, :2].sum()
        if t < start:
            continue
        cps = offline_detect(x[: t + 1], "pelt", pen_mult=pen_mult)
        if cps:
            c = cps[-1]
            out[t, 0] = t - c
            out[t, 1] = np.mean(x[c: t + 1]) - np.mean(x[max(0, c - 4): c])
        else:
            out[t, 0], out[t, 1] = 24, 0.0
    return out


def asof_features(Y: np.ndarray, groups: np.ndarray, pen_mult: float = 12, hazard: float = 0.05, n_jobs: int = 8) -> np.ndarray:
    """S × T × 3 массив признаков детектора, вычисленных строго по прошлым данным."""
    from joblib import Parallel, delayed

    X = relative_index(Y, groups)
    res = Parallel(n_jobs=n_jobs, batch_size=128)(delayed(_asof_one)(X[i], pen_mult, hazard) for i in range(len(X)))
    return np.stack(res)
