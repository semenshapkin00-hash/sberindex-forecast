"""Загрузка YAML-конфигурации с возможностью переопределения параметров из командной строки."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, other: dict) -> dict:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_path(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value)


class Config(dict):
    """dict с доступом через точку и путями относительно корня проекта."""

    def __getattr__(self, item):
        v = self[item]
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        p = p if p.is_absolute() else ROOT / p
        p.mkdir(parents=True, exist_ok=True) if key != "ca_bundle" else p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    if path and Path(path).resolve() != (ROOT / "configs" / "default.yaml").resolve():
        _deep_update(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_path(cfg, key, value)
    return Config(copy.deepcopy(cfg))


def cli_config(description: str, extra=None) -> tuple[Config, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML-файл конфигурации")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="переопределить параметр, напр. backtest.sample_series=300")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return load_config(args.config, args.set), args
