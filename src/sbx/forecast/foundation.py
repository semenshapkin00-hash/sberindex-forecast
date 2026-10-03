"""Фундаментальные модели временных рядов: Chronos-Bolt и Chronos-2 (Amazon, Apache-2.0).

* Chronos-Bolt — zero-shot, только история ряда.
* Chronos-2 — zero-shot; в режиме `covariates=True` получает:
    - длинную историю общероссийского ряда СберИндекса той же категории (2019+) как
      прошлую ковариату; таргет МО слева дополнен пропусками до 2019 г. — так модель
      «видит» многолетнюю сезонность, которой нет в 24-месячном ряду МО;
    - известные заранее ковариаты: национальный сезонный фактор (оценён по данным ≤ T),
      число рабочих и праздничных дней месяца;
    - индекс новостных шоков субъекта (прошлая ковариата), если он построен;
  и cross_learning: ряды одного субъекта и категории прогнозируются совместно.
* Chronos-2 fine-tuned — дообучение (LoRA) на окнах, целиком лежащих до точки прогноза.

Точечный прогноз — медиана (оптимальна для MAE); квантили 0.1/0.9 сохраняются для
детектора шоков по выходу за прогнозный интервал.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .base import Context, Model, shift_period

QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]


def _sa_inputs(ctx: Context, o: int, H: int, sa: bool, end: int | None = None):
    """Лог-история (опционально очищенная от национальной сезонности) и сезонная добавка на будущее."""
    end = o if end is None else end
    logY = np.log(ctx.Y[:, : end + 1])
    if not sa:
        return logY, np.zeros((ctx.S, H))
    origin = ctx.periods[o]
    hist = ctx.periods[: end + 1]
    fut = [shift_period(ctx.periods[end], h) for h in range(1, H + 1)]
    return logY - ctx.nat_seasonal_matrix(origin, hist), ctx.nat_seasonal_matrix(origin, fut)


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosBolt(Model):
    name = "Chronos-Bolt"

    def __init__(self, cfg: dict, deseasonalize: bool = False, name: str | None = None):
        self.cfg, self.sa = cfg, deseasonalize
        self.pipe = None
        if name:
            self.name = name

    def _load(self):
        if self.pipe is None:
            from chronos import BaseChronosPipeline
            self.pipe = BaseChronosPipeline.from_pretrained(self.cfg["model_id"], device_map=_device(self.cfg.get("device", "auto")),
                                                            torch_dtype=torch.float32)
        return self.pipe

    def predict(self, ctx: Context, o: int, H: int):
        pipe = self._load()
        logY, fut = _sa_inputs(ctx, o, H, self.sa)
        ctxs = [torch.tensor(logY[i], dtype=torch.float32) for i in range(ctx.S)]
        qs, outs = [], []
        bs = self.cfg.get("batch_size", 256)
        for b in range(0, len(ctxs), bs):
            q, _ = pipe.predict_quantiles(ctxs[b: b + bs], prediction_length=H, quantile_levels=QUANTILES)
            qs.append(q.float().cpu().numpy())
        q = np.exp(np.concatenate(qs, axis=0) + fut[:, :, None])  # S × H × Q
        return {"mean": q[:, :, QUANTILES.index(0.5)], **{f"q{a}": q[:, :, i] for i, a in enumerate(QUANTILES) if a != 0.5}}


class Chronos2(Model):
    name = "Chronos-2"

    def __init__(self, cfg: dict, covariates: bool = False, name: str | None = None, finetune: bool = False,
                 deseasonalize: bool = False):
        self.cfg, self.cov, self.finetune, self.sa = cfg, covariates, finetune, deseasonalize
        if name:
            self.name = name
        self.pipe = None
        self._ft_cache: dict[int, object] = {}

    def _load(self):
        if self.pipe is None:
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained(self.cfg["model_id"], device_map=_device(self.cfg.get("device", "auto")))
        return self.pipe

    # ------------------------------------------------------------------ входы
    def _inputs(self, ctx: Context, o: int, H: int, end: int | None = None) -> list[dict]:
        """Входы модели для точки o (история по индекс end включительно; по умолчанию end=o)."""
        end = o if end is None else end
        origin = ctx.periods[o]
        logY, self._fut_seas = _sa_inputs(ctx, o, H, self.sa, end)
        if not self.cov:
            return [{"target": logY[i]} for i in range(ctx.S)]
        from ..data import NATIONAL_TYPE
        nat = ctx.national.set_index("period")
        first = "2019-01"
        hist_periods = [str(p) for p in pd.period_range(first, ctx.periods[end], freq="M")]
        fut_periods = [shift_period(ctx.periods[end], h) for h in range(1, H + 1)]
        pad = len(hist_periods) - (end + 1)
        seas_h = ctx.nat_seasonal_matrix(origin, hist_periods)
        seas_f = ctx.nat_seasonal_matrix(origin, fut_periods)
        cal_h = nat.reindex(hist_periods)
        cal_f = nat.reindex(fut_periods)
        news = None
        if ctx.news is not None and "news_shock" in ctx.news:
            nw = ctx.news[(ctx.news.period <= origin)]
            nw = nw[nw.territory_id.isna()] if "territory_id" in nw else nw
            news = nw.pivot_table(index="region_key", columns="period", values="news_shock").reindex(columns=hist_periods).fillna(0)
        out = []
        cats = ctx.meta.category.to_numpy()
        regs = ctx.meta.region_key.to_numpy()
        nat_cols = {c: np.log(nat.reindex(hist_periods)[f"nat_spend::{t}"].to_numpy(float)) for c, t in NATIONAL_TYPE.items()}
        if self.sa:  # ковариату тоже очищаем от сезонности, чтобы она несла тренд/шоки
            seas_cat = {c: ctx.nat_seasonal_matrix(origin, hist_periods)[np.argmax(cats == c)] for c in nat_cols if (cats == c).any()}
            nat_cols = {c: v - seas_cat.get(c, 0) for c, v in nat_cols.items()}
        for i in range(ctx.S):
            target = np.concatenate([np.full(pad, np.nan), logY[i]])
            past = {"nat_spend": nat_cols[cats[i]], "nat_seas": seas_h[i],
                    "workdays": cal_h.n_workdays.to_numpy(float), "holidays": cal_h.n_holidays.to_numpy(float)}
            fut = {"nat_seas": seas_f[i], "workdays": cal_f.n_workdays.to_numpy(float), "holidays": cal_f.n_holidays.to_numpy(float)}
            if news is not None:
                past["news_shock"] = news.reindex([regs[i]]).fillna(0).to_numpy(float)[0] if regs[i] in news.index else np.zeros(len(hist_periods))
            out.append({"target": target, "past_covariates": past, "future_covariates": fut})
        return out

    def _predict_inputs(self, pipe, inputs: list[dict], H: int, order: np.ndarray) -> np.ndarray:
        bs = self.cfg.get("batch_size", 256)
        res = [None] * len(inputs)
        cross = bool(self.cov and self.cfg.get("cross_learning", True))
        for b in range(0, len(order), bs):
            idx = order[b: b + bs]
            q, _ = pipe.predict_quantiles([inputs[i] for i in idx], prediction_length=H, quantile_levels=QUANTILES,
                                          cross_learning=cross)
            for k, i in enumerate(idx):
                res[i] = q[k].float().cpu().numpy()   # 1 × H × Q
        return np.exp(np.stack([r[0] for r in res]))  # S × H × Q (сезонность добавляет вызывающий)

    def predict(self, ctx: Context, o: int, H: int):
        pipe = self._load()
        if self.finetune:
            pipe = self._finetuned(ctx, o)
        inputs = self._inputs(ctx, o, H)
        # группировка для cross_learning: ряды одного субъекта и категории — в одном батче
        order = np.lexsort((ctx.meta.region_key.to_numpy(), ctx.meta.category.to_numpy()))
        q = self._predict_inputs(pipe, inputs, H, order) * np.exp(self._fut_seas)[:, :, None]
        last = ctx.Y[:, [o]][:, :, None]
        q = np.clip(q, last / 5, last * 5)   # страховка от численных выбросов (не влияет на нормальные прогнозы)
        return {"mean": q[:, :, QUANTILES.index(0.5)], **{f"q{a}": q[:, :, i] for i, a in enumerate(QUANTILES) if a != 0.5}}

    # ------------------------------------------------------------------ дообучение
    def _finetuned(self, ctx: Context, o: int):
        """LoRA-дообучение на данных ≤ origin. Модель переобучается раз в `refit_every` точек."""
        every = self.cfg.get("refit_every", 6)
        first = min(self._ft_cache) if self._ft_cache else o
        key = first + ((o - first) // every) * every
        if key in self._ft_cache:
            return self._ft_cache[key]
        import copy

        from chronos.chronos2.preprocess import from_list_of_dicts

        base = copy.deepcopy(self._load())   # LoRA-обёртка меняет модель на месте — работаем с копией
        H_ft = self.cfg.get("ft_prediction_length", 6)
        train = self._inputs(ctx, key, H_ft)   # вся история до точки key; окна нарезаются внутри fit
        known = list(train[0].get("future_covariates", {}).keys())
        for d in train:
            d.pop("future_covariates", None)
        prepared = from_list_of_dicts(train, H_ft, known_covariates_names=known or None)
        if _device(self.cfg.get("device", "auto")) == "mps":
            # SDPA на MPS не поддерживает dropout при обучении — отключаем его (LoRA и так регуляризует)
            base.model.config.dropout_rate = 0.0
            for mod in base.model.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.p = 0.0
                elif isinstance(getattr(mod, "dropout", None), float):
                    mod.dropout = 0.0
        ft = base.fit(prepared, prediction_length=H_ft, finetune_mode="lora",
                      learning_rate=self.cfg.get("ft_lr", 1e-4), num_steps=self.cfg.get("ft_steps", 300),
                      batch_size=self.cfg.get("ft_batch_size", 64), min_past=6,
                      output_dir=self.cfg.get("ft_output_dir", "results/chronos2_ft"), report_to="none")
        self._ft_cache[key] = ft
        return ft


def precompute_fm_features(ctx: Context, cfg: dict, path, first_idx: int = 2, max_h: int = 12,
                           reuse: list | None = None) -> pd.DataFrame:
    """Прогнозы Chronos-2-SA + ковариаты из исторических точек t (zero-shot, только данные ≤ t).

    Используются как признаки LightGBM (стэкинг): fm = log(медиана/y_t), fm_width = log(q90/q10).
    Если переданы файлы уже посчитанных прогнозов этой модели (reuse: бэктест и валидационные
    точки), признаки собираются из них без повторного расчёта; иначе модель запускается
    из каждой точки t ≥ first_idx. Кэш: parquet с ключами (territory_id, category, origin_period, h).
    """
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    files = [Path(p) for p in (reuse or []) if Path(p).exists()]
    if files:
        for p in files:
            f = pd.read_parquet(p)
            y0 = ctx.Y[f.s.to_numpy(), f.origin.to_numpy()]
            frames.append(pd.DataFrame({"territory_id": ctx.meta.territory_id.to_numpy()[f.s], "category": ctx.meta.category.to_numpy()[f.s],
                                        "origin_period": np.array(ctx.periods)[f.origin], "h": f.h.to_numpy(),
                                        "fm": np.log(f.yhat.to_numpy() / y0), "fm_width": np.log(f["q0.9"].to_numpy() / f["q0.1"].to_numpy())}))
    else:
        model = Chronos2(cfg, covariates=True, deseasonalize=True)
        T = len(ctx.periods)
        for t in range(first_idx, T - 1):
            res = model.predict(ctx.masked(t), t, max_h)
            med, lo, hi = res["mean"], res["q0.1"], res["q0.9"]
            y0 = ctx.Y[:, [t]]
            frames.append(pd.DataFrame({"territory_id": np.repeat(ctx.meta.territory_id.to_numpy(), max_h),
                                        "category": np.repeat(ctx.meta.category.to_numpy(), max_h),
                                        "origin_period": ctx.periods[t], "h": np.tile(np.arange(1, max_h + 1), ctx.S),
                                        "fm": np.log(med / y0).ravel(), "fm_width": np.log(hi / lo).ravel()}))
            print(f"  FM-признаки: {ctx.periods[t]}", flush=True)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["territory_id", "category", "origin_period", "h"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out
