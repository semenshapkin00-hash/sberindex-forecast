"""Сборка аналитического датасета: панель рядов потребления + признаки МО + экзогенные ряды.

Все таблицы с временем используют строковый период 'YYYY-MM'. Ключ ряда — (territory_id, category).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .sources.rosstat import _norm_region

# соответствие категорий конкурса типам общероссийского ряда СберИндекса «Потребительские расходы»
NATIONAL_TYPE = {
    "Все категории": "Всего",
    "Продовольствие": "Продовольственные товары",
    "Общественное питание": "Общественное питание",
    "Маркетплейсы": "Непродовольственные товары",
    "Здоровье": "Услуги",
    "Транспорт": "Услуги",
}
CAT_CODE = {"Все категории": "all", "Продовольствие": "food", "Здоровье": "health",
            "Общественное питание": "horeca", "Транспорт": "transport", "Маркетплейсы": "market"}


def _sber_period(s: pd.Series) -> pd.Series:
    """Даты API СберИндекса — полночь по Москве в UTC (напр. 2018-11-30T21:00Z = 01.12.2018)."""
    return (pd.to_datetime(s, utc=True) + pd.Timedelta(hours=3)).dt.strftime("%Y-%m")


def load_consumption(raw: Path) -> pd.DataFrame:
    c = pd.read_parquet(Path(raw) / "hackathonlicence" / "consumption.parquet")
    c = c.rename(columns={"date": "period"})
    c["value"] = c["value"].astype(float)
    c["territory_id"] = c["territory_id"].astype(int)
    return c[["territory_id", "category", "period", "value"]].sort_values(["territory_id", "category", "period"]).reset_index(drop=True)


def load_municipal_dict(raw: Path) -> pd.DataFrame:
    """Последняя версия каждого territory_id + список всех его ОКТМО (для сопоставления)."""
    d = pd.read_excel(Path(raw) / "municipal_dict" / "t_dict_municipal_districts.xlsx")
    d["oktmo8"] = d.oktmo.astype(str).str.replace("-", "").str[:8]
    all_oktmo = d.groupby("territory_id").oktmo8.apply(list).rename("oktmo8_all")
    last = d.sort_values(["territory_id", "year_to", "year_from"]).groupby("territory_id").tail(1)
    first = d.sort_values(["territory_id", "year_from"]).groupby("territory_id").head(1).set_index("territory_id")
    last = last.set_index("territory_id").join(all_oktmo)
    last["name_2023"] = first.municipal_district_name  # название на начало периода данных
    last["region_key"] = last.region_name.map(_norm_region)
    return last.reset_index()


FEDERAL_CITY_COORDS = {"москва": (55.7558, 37.6173), "санкт": (59.9386, 30.3141), "севастополь": (44.6167, 33.5254)}


def fill_coordinates(md: pd.DataFrame, lat: str = "municipal_district_center_lat", lon: str = "municipal_district_center_lon") -> pd.DataFrame:
    """Внутригородские территории городов федерального значения не имеют центра в справочнике —
    берём координаты города; прочие пропуски — среднее по субъекту."""
    md = md.copy()
    for key, (la, lo) in FEDERAL_CITY_COORDS.items():
        m = (md.region_key == key) & md[lat].isna()
        md.loc[m, lat], md.loc[m, lon] = la, lo
    md[lat] = md[lat].fillna(md.groupby("region_key")[lat].transform("mean"))
    md[lon] = md[lon].fillna(md.groupby("region_key")[lon].transform("mean"))
    return md


def _name_key(name: str) -> str:
    stop = {"городской", "округ", "город", "муниципальный", "район", "г", "поселок", "пгт", "закрытое",
            "административно", "территориальное", "образование", "зато", "внутригородская", "территория",
            "федерального", "значения", "муниципальное", "сельское", "городское", "поселение", "рабочий"}
    toks = [w for w in re.findall(r"[а-яё]+", str(name).lower().replace("ё", "е")) if w not in stop]
    return " ".join(toks)


def static_features(raw: Path, external: Path, ids: np.ndarray) -> pd.DataFrame:
    md = load_municipal_dict(raw)
    md = fill_coordinates(md[md.territory_id.isin(ids)])
    # численность населения: сначала по любому ОКТМО территории, затем по (субъект, название)
    pop = pd.read_parquet(Path(external) / "rosstat" / "population_mo.parquet")
    pop["region_key"] = pop.region.map(_norm_region)
    pop["name_key"] = pop.name.map(_name_key)
    by_code = pop.set_index("oktmo8")
    cols = ["population_2023", "population_2024", "population_2025", "urban_population_2024"]
    rows = []
    for r in md.itertuples():
        hit = None
        for code in r.oktmo8_all[::-1]:
            if code in by_code.index:
                hit = by_code.loc[code]
                break
        if hit is None:
            cand = pop[(pop.region_key == r.region_key) & (pop.name_key == _name_key(r.municipal_district_name))]
            hit = cand.iloc[0] if len(cand) else None
        rows.append({c: (hit[c] if hit is not None else np.nan) for c in cols})
    md = pd.concat([md.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    md["population"] = md.population_2023.fillna(md.population_2024).fillna(md.population_2025)
    md["urban_share"] = md.urban_population_2024 / md.population_2024
    md["log_population"] = np.log(md.population)
    # моногорода (категория социально-экономического положения: 1 стабильно … 3 сложно)
    mono = pd.read_parquet(Path(external) / "rosstat" / "monotowns.parquet")
    mono["name_key"] = mono.mo_name.map(_name_key)
    mono = mono.groupby(["region_key", "name_key"]).monotown_status.max()
    md["monotown_status"] = [mono.get((rk, _name_key(n)), 0) for rk, n in zip(md.region_key, md.municipal_district_name)]
    ma = pd.read_parquet(Path(raw) / "hackathonlicence" / "market_access.parquet")
    md = md.merge(ma, on="territory_id", how="left")
    md["is_capital"] = (md.municipal_district_status == "административный_центр_субъекта").astype(int)
    md["is_zato"] = (md.municipal_district_status == "ЗАТО").astype(int)
    md["mo_type"] = md.municipal_district_type.map({"городской округ": 0, "муниципальный район": 1, "муниципальный округ": 2,
                                                    "внутригородская территория города федерального значения": 3}).fillna(1).astype(int)
    keep = ["territory_id", "municipal_district_name", "name_2023", "region_name", "region_code", "region_key", "oktmo",
            "municipal_district_center_lat", "municipal_district_center_lon", "population", "log_population",
            "urban_share", "monotown_status", "market_access", "is_capital", "is_zato", "mo_type"]
    return md[keep].rename(columns={"municipal_district_center_lat": "lat", "municipal_district_center_lon": "lon",
                                    "municipal_district_name": "mo_name"})


def road_neighbors(raw: Path, ids: np.ndarray, k: int = 8, max_km: float = 300.0) -> pd.DataFrame:
    """k ближайших по автодорогам МО (из connection.parquet), не дальше max_km."""
    n = pd.read_parquet(Path(raw) / "hackathonlicence" / "connection.parquet", filters=[("type", "==", "highway")])
    n = n[n.territory_id_x.isin(ids) & n.territory_id_y.isin(ids) & (n.territory_id_x != n.territory_id_y)]
    both = pd.concat([n[["territory_id_x", "territory_id_y", "distance"]],
                      n.rename(columns={"territory_id_x": "territory_id_y", "territory_id_y": "territory_id_x"})[["territory_id_x", "territory_id_y", "distance"]]])
    both = both[both.distance <= max_km].sort_values(["territory_id_x", "distance"])
    nb = both.groupby("territory_id_x").head(k).rename(columns={"territory_id_x": "territory_id", "territory_id_y": "neighbor_id"})
    nb["weight"] = 1.0 / (1.0 + nb.distance / 50.0)
    return nb.reset_index(drop=True)


def regional_exog(external: Path) -> pd.DataFrame:
    """Помесячные показатели субъектов: ИПЦ (м/м, %), зарплата (г/г), задолженность по зарплате."""
    ex = Path(external) / "rosstat"
    cpi = pd.read_parquet(ex / "cpi_regions.parquet").drop_duplicates(["region_key", "period"], keep="last")
    cpi = cpi[["region_key", "period", "cpi_total", "cpi_food", "cpi_nonfood", "cpi_services"]]
    w = pd.read_parquet(ex / "wages_regions.parquet").drop_duplicates(["region_key", "period"], keep="last")
    w = w.sort_values("period")
    w["wage_yoy"] = w.groupby("region_key").wage.transform(lambda s: np.log(s) - np.log(s.shift(12)))
    a = pd.read_parquet(ex / "wage_arrears.parquet").drop_duplicates(["region_key", "period"], keep="last")
    out = cpi.merge(w[["region_key", "period", "wage", "wage_yoy"]], on=["region_key", "period"], how="outer")
    out = out.merge(a[["region_key", "period", "wage_arrears"]], on=["region_key", "period"], how="left")
    out = out.sort_values(["region_key", "period"])
    # накопленная инфляция (индекс) — для дефлирования
    out["cpi_index"] = out.groupby("region_key").cpi_total.transform(lambda s: np.exp(np.log(s / 100).fillna(0).cumsum()))
    return out


def national_exog(external: Path) -> pd.DataFrame:
    """Общероссийские ряды: расходы СберИндекса по типам (с 12.2018), ключевая ставка, календарь."""
    ex = Path(external)
    cs = pd.read_parquet(ex / "sberindex" / "consumer-spending.parquet")
    cs["period"] = _sber_period(cs.period)
    cs = cs.pivot_table(index="period", columns="type", values="value", aggfunc="last")
    cs.columns = [f"nat_spend::{c}" for c in cs.columns]
    kr = pd.read_parquet(ex / "key_rate.parquet").set_index("period")
    cal = pd.read_parquet(ex / "work_calendar.parquet").set_index("period")
    out = cs.join(kr, how="outer").join(cal, how="outer").reset_index().rename(columns={"index": "period"})
    return out.sort_values("period").reset_index(drop=True)


def national_seasonal_factors(nat: pd.DataFrame, upto: str, exclude_years=(2020,)) -> pd.DataFrame:
    """Сезонные факторы по календарным месяцам для каждого типа расходов по данным до `upto`.

    Фактор = медиана лог-отклонений месяца от центрированного 12-мес. скользящего среднего
    (классическая декомпозиция) по наблюдениям не позже `upto` — без заглядывания вперёд.
    2020 г. исключается: локдаун апреля–мая искажает сезонный профиль.
    """
    rows = []
    hist = nat[nat.period <= upto]
    for col in [c for c in nat.columns if c.startswith("nat_spend::")]:
        s = np.log(hist.set_index("period")[col].dropna())
        if len(s) < 25:
            continue
        trend = s.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
        dev = (s - trend).dropna()
        idx = pd.PeriodIndex(dev.index, freq="M")
        dev = dev[~np.isin(idx.year, exclude_years)]
        f = dev.groupby(pd.PeriodIndex(dev.index, freq="M").month).median()
        f = f - f.mean()
        for m, v in f.items():
            rows.append({"type": col.split("::")[1], "month": int(m), "seas": float(v)})
    return pd.DataFrame(rows)


def news_region_monthly(processed: Path) -> pd.DataFrame | None:
    p = Path(processed) / "news_region_monthly.parquet"
    return pd.read_parquet(p) if p.exists() else None


def weather_monthly(external: Path) -> pd.DataFrame | None:
    p = Path(external) / "weather.parquet"
    if not p.exists():
        return None
    w = pd.read_parquet(p)
    keep = ["territory_id", "period", "t2m", "t2m_anom", "prectotcorr_anom", "snodp", "t2m_min_anom"]
    return w[[c for c in keep if c in w]]
