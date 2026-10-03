"""Шаг 4. Геопривязка и классификация заголовков, месячные новостные индексы.

    python scripts/04_build_news_features.py

Вход:  data/external/news/{interfax,lenta}/*.json (шаг 1, --only news)
Выход: data/processed/news_tagged.parquet        — заголовки с регионами/МО/типами событий
       data/processed/news_region_monthly.parquet — индексы по субъектам, МО и РФ × месяц
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx import news  # noqa: E402
from sbx.config import cli_config  # noqa: E402


def main():
    cfg, _ = cli_config(__doc__)
    ext, proc, raw = cfg.path("external"), cfg.path("processed"), cfg.path("raw")
    rows = []
    for src in cfg.external["news"]["sources"]:
        for p in sorted((ext / "news" / src).glob("*.json")):
            rows += json.loads(p.read_text())
    df = pd.DataFrame(rows).drop_duplicates(["source", "id"])
    df = df[(df.date >= "2022-12-01") & (df.date <= "2024-12-31")].reset_index(drop=True)
    print(f"заголовков: {len(df):,} ({df.source.value_counts().to_dict()})")

    md = pd.read_excel(raw / "municipal_dict" / "t_dict_municipal_districts.xlsx")
    ids = set(pd.read_parquet(proc / "panel.parquet").territory_id.unique())
    gaz = news.build_gazetteer(md, ids)
    tagged = news.tag_headlines(df, gaz)
    tagged.to_parquet(proc / "news_tagged.parquet")
    geo = (tagged.regions.str.len() > 0).mean()
    print(f"с геопривязкой к субъекту: {geo:.1%}; к МО: {(tagged.mos.str.len() > 0).mean():.1%}")
    print("доля по типам событий:", tagged.filter(like="ev_").mean().round(3).to_dict())

    static = pd.read_parquet(proc / "static.parquet")
    periods = [str(p) for p in pd.period_range("2022-12", "2024-12", freq="M")]
    agg = news.aggregate_monthly(tagged, periods, sorted(static.region_key.unique()))
    agg.to_parquet(proc / "news_region_monthly.parquet")
    print(f"месячных записей: {len(agg):,}")


if __name__ == "__main__":
    main()
