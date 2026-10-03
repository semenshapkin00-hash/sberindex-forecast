"""Шаг 2. Сборка аналитических таблиц в data/processed/.

    python scripts/02_build_dataset.py

Результат:
  panel.parquet          — ряды потребления (territory_id, category, period, value)
  static.parquet         — характеристики МО (население, доступность рынков, тип, моногород …)
  neighbors.parquet      — k ближайших соседей по автодорогам (из connection.parquet)
  exog_region.parquet    — ИПЦ, зарплаты, задолженность по субъектам РФ, помесячно
  exog_national.parquet  — расходы СберИндекса по РФ с 2018 г., ключевая ставка, календарь
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from sbx import cpd, data  # noqa: E402
from sbx.config import cli_config  # noqa: E402


def main():
    cfg, _ = cli_config(__doc__)
    raw, ext, out = cfg.path("raw"), cfg.path("external"), cfg.path("processed")
    panel = data.load_consumption(raw)
    ids = panel.territory_id.unique()
    panel.to_parquet(out / "panel.parquet")
    st = data.static_features(raw, ext, ids)
    st.to_parquet(out / "static.parquet")
    data.road_neighbors(raw, ids).to_parquet(out / "neighbors.parquet")
    data.regional_exog(ext).to_parquet(out / "exog_region.parquet")
    data.national_exog(ext).to_parquet(out / "exog_national.parquet")
    # признаки детектора точек изменений «на момент t» для прогнозной модели (связка CPD → прогноз)
    W = panel.pivot_table(index=["territory_id", "category"], columns="period", values="value")
    W = W[W.notna().sum(axis=1) >= cfg.data["min_obs"]]
    fc = cfg.cpd["features"]
    meta = W.index.to_frame(index=False).merge(st, on="territory_id", how="left")
    F = cpd.asof_features(W.to_numpy(), cpd.peer_groups(meta),
                          fc["pelt_pen_mult"], fc["bocpd_hazard"], cfg.project["n_jobs"])
    long = W.index.to_frame(index=False).loc[np.repeat(np.arange(len(W)), W.shape[1])].reset_index(drop=True)
    long["period"] = np.tile(W.columns.to_numpy(), len(W))
    long[["cp_age", "cp_mag", "cp_prob"]] = F.reshape(-1, 3)
    long.to_parquet(out / "cpd_asof.parquet")
    print(f"панель: {len(panel):,} наблюдений, {panel.groupby(['territory_id', 'category']).ngroups:,} рядов, {len(ids)} МО")
    print(f"численность населения найдена для {st.population.notna().mean():.1%} МО")
    print(f"моногорода: {(st.monotown_status > 0).sum()}")


if __name__ == "__main__":
    main()
