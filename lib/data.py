import logging
import os
import pathlib
import typing

from lib.models import Dataset

logger = logging.getLogger(__name__)


def get_data_path() -> pathlib.Path:
    path = pathlib.Path(os.path.dirname(__file__)).joinpath("..").joinpath("data").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_tables_path(dataset: Dataset) -> pathlib.Path:
    match dataset:
        case Dataset.AITQA:
            return get_data_path() / "AITQA" / "modified_data" / "contextualized_tables.jsonl"
        case Dataset.NQTABLES:
            return get_data_path() / "NQTables" / "tables" / "tables.jsonl"
        case _ as dataset:
            typing.assert_never(dataset)


def get_queries_path(dataset: Dataset, subset: str) -> pathlib.Path:
    match dataset:
        case Dataset.AITQA:
            return get_data_path() / "AITQA" / "modified_data" / f"{subset}_questions.jsonl"
        case Dataset.NQTABLES:
            return get_data_path() / "NQTables" / "classified" / f"{subset}.jsonl"
        case _ as dataset:
            typing.assert_never(dataset)
