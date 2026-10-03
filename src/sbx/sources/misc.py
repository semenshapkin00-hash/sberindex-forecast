"""Прочие открытые источники: ключевая ставка ЦБ РФ, производственный календарь, погода."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

UA = "Mozilla/5.0 (research; sberindex-forecast)"


def key_rate(start: str = "01.01.2018", end: str = "31.12.2026", url: str = "https://www.cbr.ru/hd_base/KeyRate/") -> pd.DataFrame:
    """Ключевая ставка ЦБ РФ (дневные значения) -> среднее за месяц и значение на конец месяца."""
    r = requests.get(url, params={"UniDbQuery.Posted": "True", "UniDbQuery.From": start, "UniDbQuery.To": end},
                     headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    table = BeautifulSoup(r.text, "lxml").find("table", class_="data")
    rows = []
    for tr in table.find_all("tr")[1:]:
        td = [c.get_text(strip=True) for c in tr.find_all("td")]
        if len(td) >= 2:
            rows.append((pd.to_datetime(td[0], dayfirst=True), float(td[1].replace(",", "."))))
    daily = pd.DataFrame(rows, columns=["date", "key_rate"]).sort_values("date")
    daily["period"] = daily.date.dt.to_period("M")
    return daily.groupby("period").key_rate.agg(key_rate_mean="mean", key_rate_eom="last").reset_index()


def work_calendar(years: list[int], url: str = "https://isdayoff.ru/api/getdata") -> pd.DataFrame:
    """Производственный календарь РФ: число рабочих/выходных дней в месяце (isdayoff.ru)."""
    rows = []
    for y in years:
        r = requests.get(url, params={"year": y, "cc": "ru"}, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        flags = r.text.strip()
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        assert len(flags) == len(days), "isdayoff вернул неожиданный формат"
        df = pd.DataFrame({"date": days, "off": [c != "0" for c in flags]})
        df["period"] = df.date.dt.to_period("M")
        df["weekday"] = df.date.dt.weekday
        g = df.groupby("period")
        out = pd.DataFrame({
            "n_days": g.size(),
            "n_workdays": g.off.apply(lambda s: int((~s).sum())),
            "n_offdays": g.off.sum().astype(int),
            # «лишние» выходные сверх обычных суббот/воскресений (праздники)
            "n_holidays": g.apply(lambda z: int((z.off & (z.weekday < 5)).sum()), include_groups=False),
            "n_fridays": g.weekday.apply(lambda s: int((s == 4).sum())),
        }).reset_index()
        rows.append(out)
        time.sleep(0.3)
    return pd.concat(rows, ignore_index=True)


def weather(points: pd.DataFrame, start_year: int, end_year: int, parameters: list[str], cache_dir: Path,
            url: str = "https://power.larc.nasa.gov/api/temporal/monthly/point", workers: int = 4) -> pd.DataFrame:
    """Помесячная погода NASA POWER (MERRA-2, открытые данные) для центров МО.

    points: DataFrame[territory_id, lat, lon]. Каждая точка кэшируется отдельным JSON.
    Возвращает длинную таблицу territory_id × period с исходными значениями и аномалиями
    относительно среднего по тому же календарному месяцу за годы до 2023 г.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pts = points.dropna(subset=["lat", "lon"]).drop_duplicates("territory_id").reset_index(drop=True)

    def fetch(row) -> pd.DataFrame:
        cache = cache_dir / f"power_{row.territory_id}_{start_year}_{end_year}.json"
        if not cache.exists():
            params = {"parameters": ",".join(parameters), "community": "AG", "latitude": f"{row.lat:.4f}",
                      "longitude": f"{row.lon:.4f}", "start": start_year, "end": end_year, "format": "JSON"}
            for attempt in range(5):
                try:
                    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=120)
                    if r.status_code == 429:
                        time.sleep(30 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    cache.write_text(r.text)
                    break
                except requests.RequestException:
                    time.sleep(10 * (attempt + 1))
            time.sleep(0.3)
        if not cache.exists():
            return pd.DataFrame()
        par = json.loads(cache.read_text())["properties"]["parameter"]
        df = pd.DataFrame(par)
        df = df[~df.index.str.endswith("13")]          # 13-й «месяц» — среднее за год
        df.index = pd.PeriodIndex([f"{i[:4]}-{i[4:]}" for i in df.index], freq="M")
        df = df.replace(-999.0, np.nan)
        df["territory_id"] = row.territory_id
        return df.rename_axis("period").reset_index()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = list(tqdm(ex.map(fetch, pts.itertuples(index=False)), total=len(pts), desc="weather"))
    d = pd.concat([f for f in frames if len(f)], ignore_index=True)
    d.columns = [c.lower() for c in d.columns]
    d["month"] = d.period.dt.month
    base = d[d.period.dt.year < 2023].groupby(["territory_id", "month"])[[p.lower() for p in parameters]].mean()
    d = d.merge(base.add_suffix("_clim").reset_index(), on=["territory_id", "month"], how="left")
    for p in parameters:
        p = p.lower()
        d[f"{p}_anom"] = d[p] - d[f"{p}_clim"]
    d["period"] = d.period.astype(str)
    return d.drop(columns=["month"] + [f"{p.lower()}_clim" for p in parameters])
