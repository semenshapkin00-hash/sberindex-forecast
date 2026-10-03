"""Шаг 1. Загрузка всех данных из открытых источников.

    python scripts/01_download_data.py                # всё
    python scripts/01_download_data.py --only sberindex rosstat
    python scripts/01_download_data.py --only news    # долго (~1–1.5 ч), кэшируется по дням

Источники:
  sberindex   — датасет конкурса (zip), справочник МО (rar), открытые датасеты sberindex.ru
  rosstat     — ИПЦ, зарплаты, задолженность по зарплате, численность населения МО, моногорода
  cbr         — ключевая ставка ЦБ РФ
  calendar    — производственный календарь (isdayoff.ru)
  weather     — погода NASA POWER для центров МО
  news        — заголовки новостей Интерфакс и Lenta.ru (суточные архивы)
"""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import certifi
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbx.config import ROOT, cli_config  # noqa: E402
from sbx import data  # noqa: E402
from sbx.sources import misc, news, rosstat  # noqa: E402
from sbx.sources.sberindex_api import SberIndexAPI  # noqa: E402

STEPS = ["sberindex", "rosstat", "cbr", "calendar", "weather", "news"]


def build_ca_bundle(cfg) -> Path:
    """Сайты Сбера и Росстата используют сертификаты НУЦ Минцифры, а sberindex.ru не отдаёт
    промежуточный сертификат. Собираем bundle: certifi + публичные корневые/промежуточные
    сертификаты из папки certs/ — проверка TLS остаётся включённой."""
    target = cfg.path("ca_bundle")
    parts = [Path(certifi.where()).read_text()]
    parts += [p.read_text() for p in sorted((ROOT / "certs").glob("*.pem")) if p.name != target.name]
    target.write_text("\n".join(parts))
    return target


def _unzip_cp866(blob: bytes, out_dir: Path) -> None:
    """Имена файлов в архиве конкурса записаны в cp866 без UTF-8 флага."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("cp866")
                except UnicodeError:
                    pass
            dest = out_dir / name
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))


def step_sberindex(cfg, ca):
    raw = cfg.path("raw")
    hz = raw / "hackathonlicence" / "consumption.parquet"
    if not hz.exists():
        print("· датасет конкурса")
        r = requests.get(cfg.data["hackathon_zip_url"], verify=str(ca), timeout=600)
        r.raise_for_status()
        _unzip_cp866(r.content, raw)
    md = raw / "municipal_dict" / "t_dict_municipal_districts.xlsx"
    if not md.exists():
        print("· справочник муниципальных образований")
        md.parent.mkdir(parents=True, exist_ok=True)
        rar = md.parent / "t_dict_municipal.rar"
        r = requests.get(cfg.data["municipal_dict_url"], verify=str(ca), timeout=600)
        r.raise_for_status()
        rar.write_bytes(r.content)
        # bsdtar (libarchive) есть в macOS и большинстве Linux; альтернатива — unar/7z
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(md.parent), md.name], check=True)
    out = cfg.path("external") / "sberindex"
    out.mkdir(parents=True, exist_ok=True)
    api = SberIndexAPI(ca)
    for name in cfg.sberindex_api["datasets"]:
        target = out / f"{name}.parquet"
        if target.exists():
            continue
        print(f"· sberindex.ru: {name}")
        df = api.dataset(name)
        df.to_parquet(target)
        print(f"  {len(df):,} строк")


def step_rosstat(cfg, ca):
    out = cfg.path("external") / "rosstat"
    paths = rosstat.download(out, ca)
    rosstat.parse_cpi_regions(paths["cpi_regions"]).assign(period=lambda d: d.period.astype(str)).to_parquet(out / "cpi_regions.parquet")
    rosstat.parse_wages(paths["wages_regions"]).assign(period=lambda d: d.period.astype(str)).to_parquet(out / "wages_regions.parquet")
    rosstat.parse_wage_arrears(paths["wage_arrears"]).assign(period=lambda d: d.period.astype(str)).to_parquet(out / "wage_arrears.parquet")
    rosstat.parse_population_mo({y: paths[f"population_mo_{y}"] for y in (2023, 2024, 2025)}).to_parquet(out / "population_mo.parquet")
    rosstat.parse_monotowns(paths["monotowns"]).to_parquet(out / "monotowns.parquet")
    print("· Росстат: ИПЦ, зарплаты, задолженность, численность МО, моногорода")


def step_cbr(cfg, ca):
    df = misc.key_rate(url=cfg.external["key_rate_url"])
    df.assign(period=df.period.astype(str)).to_parquet(cfg.path("external") / "key_rate.parquet")
    print(f"· ЦБ РФ: ключевая ставка, {len(df)} месяцев")


def step_calendar(cfg, ca):
    df = misc.work_calendar([2022, 2023, 2024, 2025], url=cfg.external["calendar_url"])
    df.assign(period=df.period.astype(str)).to_parquet(cfg.path("external") / "work_calendar.parquet")
    print("· производственный календарь")


def step_weather(cfg, ca):
    d = data.fill_coordinates(data.load_municipal_dict(cfg.path("raw")))
    ids = pd.read_parquet(cfg.path("raw") / "hackathonlicence" / "consumption.parquet", columns=["territory_id"]).territory_id.unique()
    pts = d[d.territory_id.isin(ids)].rename(columns={"municipal_district_center_lat": "lat", "municipal_district_center_lon": "lon"})
    w = cfg.external["weather"]
    df = misc.weather(pts[["territory_id", "lat", "lon"]], w["start_year"], w["end_year"], w["parameters"],
                      cfg.path("external") / "weather_cache", url=w["url"], workers=w["workers"])
    df.to_parquet(cfg.path("external") / "weather.parquet")
    print(f"· погода: {df.territory_id.nunique()} МО")


def step_news(cfg, ca):
    n = cfg.external["news"]
    out = cfg.path("external") / "news"
    frames = [news.collect(src, str(n["start"]), str(n["end"]), out, sleep=n["sleep"]) for src in n["sources"]]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out / "headlines.parquet")
    print(f"· новости: {len(df):,} заголовков")


def main():
    cfg, args = cli_config(__doc__, lambda ap: ap.add_argument("--only", nargs="*", choices=STEPS, default=STEPS))
    ca = build_ca_bundle(cfg)
    for step in args.only:
        globals()[f"step_{step}"](cfg, ca)


if __name__ == "__main__":
    main()
