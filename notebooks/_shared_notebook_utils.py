from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
RESEARCH_CHECKPOINT_DIR = ARTIFACTS_DIR / "research_checkpoint"
CATBOOST_DIR = ARTIFACTS_DIR / "catboost"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_DIR.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size / (1024 ** 2)


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_pickle(path)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def drop_from_globals(globals_dict: dict, names: Iterable[str]) -> int:
    dropped = 0
    for name in names:
        if name in globals_dict:
            del globals_dict[name]
            dropped += 1
    gc.collect()
    return dropped


def assert_no_group_leakage(train_ids, val_ids, test_ids) -> None:
    train_set = set(pd.Series(train_ids).astype("string"))
    val_set = set(pd.Series(val_ids).astype("string"))
    test_set = set(pd.Series(test_ids).astype("string"))

    if train_set & val_set:
        raise AssertionError("Leakage detected between train and validation CSBID sets")
    if train_set & test_set:
        raise AssertionError("Leakage detected between train and test CSBID sets")
    if val_set & test_set:
        raise AssertionError("Leakage detected between validation and test CSBID sets")


def assert_class_mapping_consistency(mapping_a: dict, mapping_b: dict) -> None:
    a_norm = {str(k): int(v) for k, v in mapping_a.items()}
    b_norm = {str(k): int(v) for k, v in mapping_b.items()}
    if a_norm != b_norm:
        raise AssertionError("Class mapping mismatch between artifacts")
