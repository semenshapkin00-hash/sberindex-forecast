"""Сбор заголовков новостей из открытых суточных архивов (Интерфакс, Lenta.ru).

Сохраняем только метаданные: дату-время публикации, заголовок, рубрику и ссылку.
Полные тексты не скачиваются. Каждый день кэшируется отдельным файлом, поэтому
сбор можно прерывать и продолжать. Пути архивов разрешены robots.txt обоих сайтов.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
RU_MONTHS = {m: i for i, m in enumerate(
    ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
     "сентября", "октября", "ноября", "декабря"], start=1)}


def _get(session: requests.Session, url: str, sleep: float) -> requests.Response | None:
    for attempt in range(4):
        try:
            r = session.get(url, timeout=40)
            time.sleep(sleep)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def _interfax_day(session, d: date, sleep: float) -> list[dict]:
    rows, seen = [], set()
    for page in range(1, 12):
        suffix = "" if page == 1 else f"page_{page}"
        r = _get(session, f"https://www.interfax.ru/news/{d:%Y/%m/%d}/{suffix}", sleep)
        if r is None:
            break
        try:
            html = r.content.decode("utf-8")
        except UnicodeDecodeError:  # старые страницы архива отдаются в windows-1251
            html = r.content.decode("cp1251", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        new = 0
        for div in soup.select("div.an div[data-id]"):
            nid = div["data-id"]
            if nid in seen:
                continue
            a, t = div.find("a", href=True), div.find("span")
            h = div.find("h3")
            if not (a and h):
                continue
            seen.add(nid)
            new += 1
            href = a["href"]
            rubric = href.strip("/").split("/")[0] if href.startswith("/") else re.sub(r"https?://(www\.)?", "", href).split(".")[0]
            rows.append({"source": "interfax", "id": nid, "date": d.isoformat(),
                         "time": t.get_text(strip=True) if t else None,
                         "title": h.get_text(" ", strip=True), "rubric": rubric, "url": href})
        if new == 0 or not soup.find("a", href=re.compile(rf"/news/{d:%Y/%m/%d}/page_{page + 1}")):
            break
    return rows


def _lenta_day(session, d: date, sleep: float) -> list[dict]:
    rows, seen = [], set()
    for page in range(1, 15):
        suffix = "" if page == 1 else f"page/{page}/"
        r = _get(session, f"https://lenta.ru/news/{d:%Y/%m/%d}/{suffix}", sleep)
        if r is None:
            break
        soup = BeautifulSoup(r.text, "lxml")
        new = 0
        for li in soup.select("li.archive-page__item a.card-full-news"):
            href = li.get("href", "")
            if href in seen:
                continue
            h = li.select_one(".card-full-news__title")
            if not h:
                continue
            seen.add(href)
            new += 1
            t = li.select_one("time")
            rub = li.select_one(".card-full-news__rubric")
            tm = t.get_text(strip=True).split(",")[0] if t else None
            rows.append({"source": "lenta", "id": href, "date": d.isoformat(), "time": tm,
                         "title": h.get_text(" ", strip=True),
                         "rubric": rub.get_text(strip=True) if rub else None, "url": href})
        if new == 0:
            break
    return rows


SCRAPERS = {"interfax": _interfax_day, "lenta": _lenta_day}


def collect(source: str, start: str, end: str, out_dir: Path, sleep: float = 0.7, workers: int = 2) -> pd.DataFrame:
    """Собирает заголовки за период [start, end]; уже скачанные дни берутся из кэша."""
    out_dir = Path(out_dir) / source
    out_dir.mkdir(parents=True, exist_ok=True)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [d0 + timedelta(i) for i in range((d1 - d0).days + 1)]
    todo = [d for d in days if not (out_dir / f"{d.isoformat()}.json").exists()]
    fn = SCRAPERS[source]

    def work(d: date):
        session = requests.Session()
        session.headers["User-Agent"] = UA
        rows = fn(session, d, sleep)
        if rows:  # пустой день не кэшируем, чтобы повторить при следующем запуске
            (out_dir / f"{d.isoformat()}.json").write_text(json.dumps(rows, ensure_ascii=False))
        return len(rows)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(work, todo), total=len(todo), desc=f"news:{source}"))
    frames = [pd.DataFrame(json.loads(p.read_text())) for p in sorted(out_dir.glob("*.json"))
              if d0.isoformat() <= p.stem <= d1.isoformat()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
