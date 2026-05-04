import json
import os
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from RUL_prediction import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("yaml file is empty")
            logger.info("yaml file loaded: %s", path_to_yaml)
            return ConfigBox(content)
    except BoxValueError as exc:
        raise ValueError("yaml file is empty") from exc


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info("created directory at: %s", path)


@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info("json file saved at: %s", path)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info("json file loaded from: %s", path)
    return ConfigBox(content)


def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info("binary file saved at: %s", path)


def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logger.info("binary file loaded from: %s", path)
    return data
