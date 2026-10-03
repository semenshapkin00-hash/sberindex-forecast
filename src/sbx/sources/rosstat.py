"""Загрузка и разбор открытых таблиц Росстата (rosstat.gov.ru/storage/mediabank).

* ИПЦ по субъектам РФ, помесячно (всего / продовольствие / непродовольственные / услуги);
* среднемесячная номинальная начисленная зарплата по субъектам, помесячно;
* просроченная задолженность по зарплате по субъектам;
* численность постоянного населения по муниципальным образованиям (ТЕРСОН-МО/ОКТМО);
* список моногородов с категорией социально-экономического положения.

Имена файлов на сайте содержат месяц публикации, поэтому ссылка ищется по шаблону на
странице раздела, а не задаётся жёстко.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://rosstat.gov.ru"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]

# раздел сайта -> регулярное выражение для имени файла
FILES = {
    "cpi_regions": ("/statistics/price", r"ipc_RF_fo_sub_\d{2}-\d{4}\.xlsx"),
    "wages_regions": ("/labor_market_employment_salaries", r"tab2-zpl_\d{2}-\d{4}\.xlsx"),
    "wage_arrears": ("/labor_market_employment_salaries", r"tab11_zpl_\d{2}-\d{4}\.xlsx"),
    "population_mo_2023": (None, "BUL_MO_2023.xlsx"),
    "population_mo_2024": (None, "BUL_MO_2024.xlsx"),
    "population_mo_2025": ("/folder/12781", r"hisl_MO_01-01-2025\.xlsx"),
    "monotowns": ("/folder/12781", r"popul_monogorod_01-01-\d{4}\.xlsx"),
}


def _session(ca_bundle) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    s.verify = str(ca_bundle)
    return s


def download(out_dir: Path, ca_bundle) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _session(ca_bundle)
    pages: dict[str, str] = {}
    paths = {}
    for key, (page, pattern) in FILES.items():
        if page is None:
            fname = pattern
        else:
            if page not in pages:
                pages[page] = s.get(BASE + page, timeout=90).text
            hrefs = [a["href"] for a in BeautifulSoup(pages[page], "lxml").find_all("a", href=True)]
            cands = sorted({h.split("/")[-1] for h in hrefs if re.search(pattern, h.split("/")[-1])})
            if not cands:
                raise RuntimeError(f"Росстат: не найден файл {pattern} на странице {page}")
            fname = cands[-1]
        target = out_dir / f"{key}.xlsx"
        if not target.exists():
            r = s.get(f"{BASE}/storage/mediabank/{fname}", timeout=180)
            r.raise_for_status()
            target.write_bytes(r.content)
        paths[key] = target
    return paths


_GENERIC = {"республика", "респ", "область", "обл", "край", "кр", "автономный", "автономная", "авт",
            "округ", "ао", "г", "город", "федерации", "российской", "кроме", "без", "и", "в", "том", "числе"}
_LAT2CYR = str.maketrans("aceopxyhkmtbABCEHKMOPTXY", "асеорхункмтвАВСЕНКМОРТХУ")


def _norm_region(name: str) -> str:
    """Ключ субъекта РФ: первое содержательное слово названия.

    'Республика Саха (Якутия)' -> 'саха', 'Кемеровская область - Кузбасс' -> 'кемеровская',
    'Ханты-Мансийский авт. округ - Югра' -> 'ханты', 'г. Москва' -> 'москва'.
    """
    n = str(name).translate(_LAT2CYR).lower().replace("ё", "е")
    tokens = [w for w in re.findall(r"[а-я]+", n) if w not in _GENERIC]
    return tokens[0] if tokens else n


def parse_cpi_regions(path: Path) -> pd.DataFrame:
    """ИПЦ к предыдущему месяцу (%), листы вида '03(2024)'."""
    x = pd.ExcelFile(path)
    rows = []
    for sh in x.sheet_names:
        m = re.fullmatch(r"(\d{2})\((\d{4})\)", sh.strip())
        if not m:
            continue
        df = pd.read_excel(x, sheet_name=sh, header=None)
        period = pd.Period(f"{m.group(2)}-{m.group(1)}", "M")
        for _, r in df.iterrows():
            code, name = r.iloc[0], r.iloc[1]
            if not isinstance(name, str) or not re.match(r"^\d+$", str(code).strip()):
                continue
            vals = pd.to_numeric(r.iloc[2:6], errors="coerce").to_numpy()
            rows.append({"period": period, "okato": str(code).strip(), "region": name.strip(),
                         "cpi_total": vals[0], "cpi_food": vals[1], "cpi_nonfood": vals[2], "cpi_services": vals[3]})
    out = pd.DataFrame(rows)
    out["region_key"] = out.region.map(_norm_region)
    return out


def _parse_monthly_wide(df: pd.DataFrame, first_year: int) -> pd.DataFrame:
    """Таблица 'регион × (год, месяц)' -> длинный формат."""
    hdr_row = next(i for i in range(10) if str(df.iloc[i, 1]).strip().lower().startswith("январ"))
    months = df.iloc[hdr_row, 1:].tolist()
    periods, year = [], first_year - 1
    for m in months:
        m = str(m).strip().lower()
        if m.startswith("январ"):
            year += 1
        idx = next((k for k, name in enumerate(MONTHS) if m.startswith(name[:4])), None)
        periods.append(pd.Period(f"{year}-{idx + 1:02d}", "M") if idx is not None else None)
    rows = []
    for i in range(hdr_row + 1, len(df)):
        name = df.iloc[i, 0]
        if not isinstance(name, str) or not name.strip():
            continue
        for j, p in enumerate(periods, start=1):
            if p is None:
                continue
            v = pd.to_numeric(df.iloc[i, j], errors="coerce")
            if pd.notna(v):
                rows.append({"period": p, "region": name.strip(), "value": float(v)})
    out = pd.DataFrame(rows)
    out["region_key"] = out.region.map(_norm_region)
    return out


def parse_wages(path: Path) -> pd.DataFrame:
    x = pd.ExcelFile(path)
    sh = next(s for s in x.sheet_names if "2019" in s)
    return _parse_monthly_wide(pd.read_excel(x, sheet_name=sh, header=None), 2019).rename(columns={"value": "wage"})


def parse_wage_arrears(path: Path) -> pd.DataFrame:
    """Просроченная задолженность по зарплате на 1 число месяца, тыс. руб."""
    x = pd.ExcelFile(path)
    frames = []
    for sh in x.sheet_names:
        m = re.search(r"_(?:с )?(\d{4})$", sh.strip())
        if not m or "задолженность на 1" in sh:
            continue
        df = pd.read_excel(x, sheet_name=sh, header=None)
        try:
            frames.append(_parse_monthly_wide(df.iloc[:, :], int(m.group(1))))
        except StopIteration:
            continue
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["period", "region_key"], keep="last")
    return out.rename(columns={"value": "wage_arrears"})


def _pop_rows(df: pd.DataFrame, with_codes: bool) -> pd.DataFrame:
    rows = []
    off = 1 if with_codes else 0
    for _, r in df.iterrows():
        name = r.iloc[off]
        tot = pd.to_numeric(r.iloc[off + 1], errors="coerce")
        if not isinstance(name, str) or pd.isna(tot):
            continue
        code = str(r.iloc[0]).strip().split(".")[0] if with_codes else None
        if with_codes and not re.fullmatch(r"\d{10}|\d{15}", code or ""):
            continue
        rows.append({"code": code, "name": " ".join(name.split()), "population": float(tot),
                     "urban_population": pd.to_numeric(r.iloc[off + 2], errors="coerce"),
                     "rural_population": pd.to_numeric(r.iloc[off + 3], errors="coerce")})
    return pd.DataFrame(rows)


def _read_pop(path: Path) -> pd.DataFrame:
    x = pd.ExcelFile(path)
    sh = next(s for s in x.sheet_names if "Численность_по_МО" in s or s == "Таб_27")
    df = pd.read_excel(x, sheet_name=sh, header=None, dtype=str)
    with_codes = df.iloc[8:200, 0].astype(str).str.fullmatch(r"\d{10}(\d{5})?").mean() > 0.5
    return _pop_rows(df, with_codes)


def parse_population_mo(years: dict[int, Path]) -> pd.DataFrame:
    """Численность населения МО (на 1 января) по ОКТМО, широкая таблица population_<год>.

    Бюллетени 2024–2025 гг. содержат коды ТЕРСОН-МО; бюллетень 2023 г. — только названия,
    поэтому он сопоставляется с кодами по паре (субъект, название МО).
    """
    coded, named = {}, {}
    for year, p in sorted(years.items()):
        df = _read_pop(p)
        (coded if df.code.notna().any() else named)[year] = df
    out, regions = None, set()
    for year, df in coded.items():
        df = df[df.code.str.len() == 10].copy()
        is_region = df.code.str.endswith("00000000")
        df["region"] = df.name.where(is_region).ffill()
        regions |= set(df.name[is_region])
        df = df[~is_region].assign(oktmo8=lambda z: z.code.str[:8])
        cols = {"population": f"population_{year}", "urban_population": f"urban_population_{year}"}
        df = df[["oktmo8", "region", "name", "population", "urban_population"]].rename(columns=cols)
        out = df if out is None else out.merge(df.drop(columns=["region", "name"]), on="oktmo8", how="outer")
    for year, df in named.items():
        df["region"] = df.name.where(df.name.isin(regions)).ffill()
        df = df[~df.name.isin(regions)].drop_duplicates(["region", "name"])
        out = out.merge(df[["region", "name", "population"]].rename(columns={"population": f"population_{year}"}),
                        on=["region", "name"], how="left")
    return out.drop_duplicates("oktmo8").reset_index(drop=True)


def parse_monotowns(path: Path) -> pd.DataFrame:
    x = pd.ExcelFile(path)
    status = {"Сложное": 3, "Риски": 2, "Стабильное": 1}
    rows = []
    for sh, lvl in status.items():
        if sh not in x.sheet_names:
            continue
        df = pd.read_excel(x, sheet_name=sh, header=None)
        for _, r in df.iterrows():
            if pd.notna(r.iloc[0]) and isinstance(r.iloc[2], str):
                rows.append({"region": r.iloc[1], "mo_name": r.iloc[2], "monotown_status": lvl})
    out = pd.DataFrame(rows)
    out["region_key"] = out.region.map(_norm_region)
    return out
