"""Интеграция новостей с данными СберИндекса.

Механизм согласования (воспроизводимый, без ручной разметки):

1. Нормализация текста: токены заголовка → леммы (pymorphy3).
2. Геопривязка по газетиру, построенному из справочника МО СберИндекса:
   * субъект РФ — полное название («курский область»), собственное имя («татарстан»,
     «кузбасс»), административный центр («белгород»), устойчивые синонимы («кубань»);
   * муниципалитет — «<прилагательное> район/округ» или город-центр МО. Неоднозначные
     названия (есть в нескольких субъектах или совпадают с обычными словами) засчитываются
     только при упоминании своего субъекта в том же заголовке.
3. Классификация событий словарём лемм: безопасность, ЧС/стихия, коммунальные аварии,
   негатив/позитив в экономике, эпидемии + отраслевые темы для категорий трат
   (маркетплейсы, топливо/транспорт, лекарства/медицина, общепит, продукты).
4. Временное согласование «as-of»: новость относится к месяцу публикации (МСК); признак
   для точки прогноза T использует только новости, опубликованные не позже конца T.
   Для раннего предупреждения о шоке в месяце t+1 используются новости ≤ t.
5. Пространственное согласование: МО получает (а) собственные упоминания, (б) новости своего
   субъекта, (в) общероссийские отраслевые новости своей категории трат.
6. Нормировка: log(1+count) и «новостной сюрприз» — отклонение от собственной средней
   за предыдущие 6 месяцев (только прошлые месяцы).
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*", re.I)

# --- события: леммы (или префиксы лемм с '*') -------------------------------------------------
EVENTS = {
    "security": ["беспилотник", "бпла", "дрон", "обстрел", "обстрелять", "атака", "атаковать", "ракета", "ракетный",
                 "взрыв", "сбить", "эвакуация", "эвакуировать", "вторжение", "всу", "диверсант", "дрг",
                 "контртеррористический", "артобстрел", "миномётный", "минометный", "снаряд", "воздушный тревога",
                 "приграничный", "укрытие", "детонация", "прилёт", "прилет", "повреждение", "пострадать", "ранить"],
    "disaster": ["паводок", "наводнение", "подтопление", "подтопить", "затопить", "затопление", "дамба", "половодье",
                 "пожар", "лесной пожар", "землетрясение", "ураган", "шторм", "смерч", "ливень", "снегопад", "метель",
                 "гололёд", "гололед", "аномальный", "засуха", "град", "чрезвычайный ситуация", "режим чс", "чс",
                 "стихия", "непогода", "мороз", "жара"],
    "utility": ["отключение", "обесточить", "электроснабжение", "теплоснабжение", "водоснабжение", "блэкаут",
                "без свет", "без тепло", "без вода", "коммунальный авария", "авария на сеть", "энергосистема"],
    "econ_neg": ["банкротство", "банкрот", "закрытие", "сокращение", "увольнение", "уволить", "простой",
                 "задолженность", "долг по зарплата", "забастовка", "остановка", "дефицит", "подорожание",
                 "подорожать", "рост цена", "кризис", "убыток", "снижение спрос", "приостановить"],
    "econ_pos": ["открытие", "открыть", "запуск", "запустить", "инвестиция", "инвестор", "новый завод",
                 "рост производство", "субсидия", "выплата", "повышение зарплата", "индексация", "строительство",
                 "ввести в эксплуатация"],
    "epidemic": ["эпидемия", "вспышка", "карантин", "грипп", "коронавирус", "ковид", "covid", "орви",
                 "отравление", "кишечный инфекция", "заболеваемость"],
}
# отраслевые темы -> категория трат конкурса
TOPICS = {
    "market": ["маркетплейс", "wildberries", "вайлдберриз", "ozon", "озон", "яндекс маркет", "пвз", "пункт выдача"],
    "transport": ["бензин", "топливо", "аи-92", "аи-95", "дизель", "азс", "такси", "каршеринг", "проезд", "автобус"],
    "health": ["лекарство", "аптека", "медикамент", "больница", "поликлиника", "минздрав", "медицинский"],
    "horeca": ["ресторан", "кафе", "общепит", "фастфуд", "доставка еда", "бар"],
    "food": ["продукт", "продовольствие", "яйцо", "молоко", "картофель", "овощ", "хлеб", "сахар", "масло",
             "продуктовый", "супермаркет"],
}
SHOCK_WEIGHTS = {"security": 1.0, "disaster": 1.0, "utility": 0.7, "econ_neg": 0.5, "epidemic": 0.5}

REGION_ALIASES = {
    "краснодарский": ["кубань"], "московский": ["подмосковье"], "санкт-петербург": ["петербург", "питер", "спб"],
    "ямало-ненецкий": ["ямал", "янао"], "ханты-мансийский": ["югра", "хмао"], "приморский": ["приморье"],
    "ставропольский": ["ставрополье"], "забайкальский": ["забайкалье"], "пермский": ["прикамье"],
    "башкортостан": ["башкирия"], "чеченский": ["чечня"], "кабардино-балкарский": ["кабардино-балкария"],
    "карачаево-черкесский": ["карачаево-черкесия"], "северный": ["алания", "северный осетия"],
    "удмуртский": ["удмуртия"], "чувашский": ["чувашия"], "тыва": ["тува"], "саха": ["якутия"],
    "кемеровский": ["кузбасс"], "москва": ["москва"], "севастополь": ["севастополь"], "крым": ["крым"],
}
GENERIC = {"республика", "область", "край", "автономный", "округ", "город", "федеральный", "значение", "г",
           "муниципальный", "район", "городской", "внутригородской", "территория", "-", "и", "кроме"}


@lru_cache(maxsize=None)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=500_000)
def lemma(word: str) -> str:
    return _morph().parse(word.lower())[0].normal_form.replace("ё", "е")


@lru_cache(maxsize=200_000)
def is_toponym(word: str) -> bool:
    p = _morph().parse(word)
    return any("Geox" in x.tag for x in p[:2])


def lemmatize(text: str) -> list[str]:
    return [lemma(t) for t in TOKEN.findall(str(text).lower())]


def _ngrams(lem: list[str], n: int = 3) -> set[str]:
    out = set()
    for k in range(1, n + 1):
        out |= {" ".join(lem[i: i + k]) for i in range(len(lem) - k + 1)}
    return out


# ------------------------------------------------------------------------- газетир
def build_gazetteer(municipal_dict: pd.DataFrame, ids: set[int]) -> dict:
    """municipal_dict: все версии МО из справочника СберИндекса (с region_key)."""
    from .sources.rosstat import _norm_region

    md = municipal_dict.copy()
    md["region_key"] = md.region_name.map(_norm_region)
    region_patterns: dict[str, set[str]] = defaultdict(set)
    region_adj: dict[str, tuple[str, str]] = {}   # «белгородский» -> (субъект, «область»)
    for name, rk in md[["region_name", "region_key"]].drop_duplicates().itertuples(index=False):
        base = re.sub(r"\(.*?\)", " ", name)
        lem = [x for x in lemmatize(base)]
        region_patterns[rk].add(" ".join(lem))
        core = [x for x in lem if x not in GENERIC]
        noun = next((x for x in lem if x in ("область", "край", "округ", "республика")), None)
        if core and noun and core[0].endswith(("ский", "цкий", "ный", "ной", "ий")):
            region_adj[core[0]] = (rk, noun)
        if core:
            # прилагательное + родовое слово («курский область») уже в полном имени;
            # собственное имя субъекта («татарстан», «адыгея») — одиночным токеном
            for c in core:
                if not c.endswith(("ский", "цкий", "ной", "ный", "ой")):
                    region_patterns[rk].add(c)
        for paren in re.findall(r"\((.*?)\)", name):
            region_patterns[rk].update(lemmatize(paren))
        for key, aliases in REGION_ALIASES.items():
            if key in lem or key == rk:
                region_patterns[rk].update(aliases)
    # административные центры субъектов
    caps = md[md.municipal_district_status == "административный_центр_субъекта"]
    for c, rk in caps[["municipal_district_center", "region_key"]].itertuples(index=False):
        city = re.sub(r"^(г|пгт|с|п|рп)\s+", "", str(c)).strip()
        if city:
            region_patterns[rk].add(" ".join(lemmatize(city)))
    # муниципалитеты
    mo_patterns: dict[str, set[tuple[int, str]]] = defaultdict(set)
    city_owner: dict[str, int] = {}   # город -> городской округ с этим городом (приоритет над районом)
    for r in md[md.territory_id.isin(ids)].itertuples(index=False):
        lem = lemmatize(re.sub(r"\(.*?\)", " ", r.municipal_district_name))
        core = [x for x in lem if x not in GENERIC]
        if not core:
            continue
        if any(x in lem for x in ("район", "округ")) and core[0].endswith(("ский", "цкий", "ный", "ной", "ий")):
            mo_patterns[f"{core[0]} район"].add((r.territory_id, r.region_key))
            mo_patterns[f"{core[0]} округ"].add((r.territory_id, r.region_key))
        city = re.sub(r"^(г|пгт|с|п|рп|ст-ца|аул|рп)\s+", "", str(r.municipal_district_center)).strip()
        if city and city != "nan":
            mo_patterns[" ".join(lemmatize(city))].add((r.territory_id, r.region_key))
        if r.municipal_district_type == "городской округ":
            mo_patterns[" ".join(core)].add((r.territory_id, r.region_key))
            city_owner[" ".join(core)] = r.territory_id
    # однозначность: имя без упоминания субъекта засчитываем, только если оно в одном субъекте
    # и является топонимом (не обычным словом)
    mo_index = {}
    for pat, cands in mo_patterns.items():
        if pat in city_owner:  # «Оренбург» -> городской округ, а не Оренбургский район с центром там же
            cands = {c for c in cands if c[0] == city_owner[pat]} or cands
        regions = {rk for _, rk in cands}
        standalone = len(regions) == 1 and (" " in pat or is_toponym(pat.split()[0]))
        mo_index[pat] = (sorted(cands), standalone)
    reg_index = defaultdict(set)
    for rk, pats in region_patterns.items():
        for p in pats:
            if p and len(p) > 2:
                reg_index[p].add(rk)
    return {"region": dict(reg_index), "mo": mo_index, "region_adj": region_adj}


def _match_keywords(grams: set[str], words: list[str]) -> int:
    return sum(1 for w in words if w in grams)


def tag_headlines(df: pd.DataFrame, gaz: dict) -> pd.DataFrame:
    """Добавляет к заголовкам: регионы, МО, типы событий, отраслевые темы."""
    rows = []
    reg_index, mo_index = gaz["region"], gaz["mo"]
    for title in df.title.astype(str):
        lem = lemmatize(title)
        grams = _ngrams(lem, 3)
        regions = set()
        for g in grams & reg_index.keys():
            regions |= reg_index[g]
        lemset = set(lem)
        for adj in lemset & gaz["region_adj"].keys():   # «Белгородской и Тульской областями»
            rk, noun = gaz["region_adj"][adj]
            if noun in lemset:
                regions.add(rk)
        mos = set()
        for g in grams & mo_index.keys():
            cands, standalone = mo_index[g]
            for tid, rk in cands:
                if standalone or rk in regions:
                    mos.add(tid)
                    regions.add(rk)
        ev = {k: _match_keywords(grams, v) for k, v in EVENTS.items()}
        tp = {k: _match_keywords(grams, v) for k, v in TOPICS.items()}
        rows.append({"regions": sorted(regions), "mos": sorted(mos), **{f"ev_{k}": int(v > 0) for k, v in ev.items()},
                     **{f"topic_{k}": int(v > 0) for k, v in tp.items()}})
    tags = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, tags], axis=1)
    out["shock_score"] = sum(out[f"ev_{k}"] * w for k, w in SHOCK_WEIGHTS.items())
    out["tone"] = out.ev_econ_pos - out.ev_econ_neg - out.ev_security - out.ev_disaster - out.ev_utility
    return out


def aggregate_monthly(tagged: pd.DataFrame, periods: list[str], region_keys: list[str]) -> pd.DataFrame:
    """Месячные индексы по субъектам (territory_id = NaN), МО и РФ (region_key='__RU__')."""
    t = tagged.copy()
    t["period"] = t.date.str[:7]
    ev_cols = [c for c in t.columns if c.startswith("ev_")]
    tp_cols = [c for c in t.columns if c.startswith("topic_")]
    val_cols = ev_cols + ["shock_score", "tone"]
    # субъекты
    reg = t[t.regions.str.len() > 0].explode("regions").rename(columns={"regions": "region_key"})
    g = reg.groupby(["region_key", "period"])
    R = g[val_cols].sum()
    R["n"] = g.size()
    R = R.reindex(pd.MultiIndex.from_product([region_keys, periods], names=["region_key", "period"])).fillna(0)
    out = pd.DataFrame(index=R.index)
    out["news_n"] = np.log1p(R.n)
    out["news_shock"] = np.log1p(R.shock_score)
    for c in ev_cols:
        out[f"news_{c[3:]}"] = np.log1p(R[c])
    out["news_tone"] = R.tone / (R.n + 5)
    # «сюрприз»: отклонение от средней за 6 предыдущих месяцев (только прошлое)
    for c in ["news_shock", "news_security", "news_disaster", "news_n"]:
        prev = out.groupby(level=0)[c].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
        out[f"{c}_surprise"] = out[c] - prev
    out = out.reset_index()
    out["territory_id"] = np.nan
    # муниципалитеты (собственные упоминания)
    mo = t[t.mos.str.len() > 0].explode("mos").rename(columns={"mos": "territory_id"})
    if len(mo):
        gm = mo.groupby(["territory_id", "period"])
        M = pd.DataFrame({"news_n_mo": np.log1p(gm.size()), "news_shock_mo": np.log1p(gm.shock_score.sum())}).reset_index()
        out = pd.concat([out, M], ignore_index=True)
    # РФ: отраслевые темы без привязки к региону -> признак для категории трат
    nat = t.groupby("period")[tp_cols + ["shock_score"]].sum().reindex(periods).fillna(0)
    N = np.log1p(nat).add_prefix("news_nat_").reset_index()
    N["region_key"] = "__RU__"
    N["territory_id"] = np.nan
    return pd.concat([out, N], ignore_index=True)
