"""Клиент публичного API сайта sberindex.ru.

Дашборды СберИндекса получают данные через POST /api/sowa: исходный GET-маршрут
(например, ``/dataset/v1/consumer-spending?filter=...``) кодируется в base64, а ответ
приходит в типизированной обёртке (``__string__<base64>``, ``__number__<x>`` и т.п.).
Модуль повторяет эту схему и возвращает обычные pandas.DataFrame.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.parse
import uuid
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://sberindex.ru/api/sowa"
_TYPED = re.compile(r"^__([a-z0-9]*)__(.*)$", re.S)


def _decode(node, hint: str | None = None):
    if isinstance(node, list):
        return [_decode(x) for x in node]
    if isinstance(node, dict) and node.get("type") == "object":
        out = {}
        for item in node["value"]:
            if item.get("type") == "longstring":
                return "".join(_decode(item["value"]))
            out[item["key"]] = _decode(item["value"], item.get("type"))
        return out
    m = _TYPED.match(str(node))
    if not m:
        return node
    kind = hint if hint and hint != "object" else m.group(1)
    value = m.group(2)
    if kind == "null":
        return None
    if kind == "number":
        return float(value)
    if kind == "boolean":
        return value == "true"
    if kind == "string":
        return base64.b64decode(value).decode("utf-8")
    return value


class SberIndexAPI:
    def __init__(self, ca_bundle: str | Path | bool = True, timeout: int = 90, pause: float = 0.3):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (research; sberindex-forecast)"
        self.verify = str(ca_bundle) if isinstance(ca_bundle, Path) else ca_bundle
        self.timeout = timeout
        self.pause = pause

    def get(self, route: str):
        body = {
            "SOWA": {
                "method": "GET",
                "route": base64.b64encode(route.encode()).decode(),
                "data": {"type": "object", "value": []},
            }
        }
        for attempt in range(4):
            try:
                # сервер требует уникальный идентификатор запроса (32 hex-символа)
                headers = {"rquid": uuid.uuid4().hex}
                r = self.session.post(API_URL, json=body, headers=headers, verify=self.verify, timeout=self.timeout)
                r.raise_for_status()
                payload = r.json()
                time.sleep(self.pause)
                return _decode(payload["SOWA"]["data"]) if "SOWA" in payload else payload
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def dataset(self, name: str, flt: dict | None = None, page: int = 1000) -> pd.DataFrame:
        """Выгружает весь датасет ``/dataset/v1/<name>`` постранично."""
        frames, offset = [], 0
        q = "" if flt is None else "filter=" + urllib.parse.quote(json.dumps(flt, separators=(",", ":"), ensure_ascii=False)) + "&"
        while True:
            res = self.get(f"/dataset/v1/{name}?{q}limit={page}&offset={offset}")
            if not isinstance(res, dict) or "data" not in res:
                break
            rows = res["data"]
            if not rows:
                break
            frames.append(pd.DataFrame(rows, columns=res["fields"]))
            if len(rows) < page:
                break
            offset += page
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
