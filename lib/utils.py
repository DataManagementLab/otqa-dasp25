import csv
import functools
import io
import json
import logging
import re
import typing

import tiktoken
from tiktoken import Encoding

from lib.data import get_tables_path
from lib.models import Dataset, TableData, QueryData, Linearization

logger = logging.getLogger(__name__)


def preprocess_cell(encoding: Encoding, cell_content: str, available_tokens_per_cell: int) -> str:
    cell_len = len(encoding.encode(cell_content))
    if cell_len > available_tokens_per_cell:
        tokens = encoding.encode(cell_content)
        truncated_tokens = tokens[:available_tokens_per_cell]
        cell_content = encoding.decode(truncated_tokens)
    return cell_content


def preprocess_table(dataset: Dataset, table: dict, llm_model_name: str, token_limit: int) -> dict:
    match dataset:
        case Dataset.AITQA:
            encoding = tiktoken.encoding_for_model(llm_model_name)
            preprocessed_table = table.copy()
            preprocessed_data_outer = []
            for data_inner in table["data"]:
                preprocessed_data_inner = [preprocess_cell(encoding, data, token_limit) for data in data_inner]
                preprocessed_data_outer.append(preprocessed_data_inner)
            preprocessed_table["data"] = preprocessed_data_outer
            return preprocessed_table
        case Dataset.NQTABLES:
            encoding = tiktoken.encoding_for_model(llm_model_name)
            preprocessed_table = table.copy()
            preprocessed_table.pop("rows", None)
            preprocessed_rows = []
            for row in table["rows"]:
                preprocessed_row = []
                for cell in row["cells"]:
                    preprocessed_cell = preprocess_cell(encoding, cell["text"], token_limit)
                    preprocessed_row.append({"text": preprocessed_cell})
                preprocessed_rows.append({"cells": preprocessed_row})
            preprocessed_table["rows"] = preprocessed_rows
            return preprocessed_table
        case _ as dataset:
            typing.assert_never(dataset)


def json_table_rep(dataset: Dataset, table: dict, row_limit: int | None) -> str:
    table_content = ""

    match dataset:
        # AIT-QA tables do not contain any irrelevant information
        case Dataset.AITQA:
            table_content = {
                "id": table.get("id", []),
                "title": table.get("title", []),
                "column_header": table.get("column_header", []),
                "row_header": table.get("row_header", [])[:row_limit],
                "data": table.get("data", [])[:row_limit],
            }
        case Dataset.NQTABLES:
            # omit JSON fields that contain the document url, alternative document urls, and
            # alternative table ids (irrelevant for the table QA task)
            table_content = {
                "id": table.get("tableId", []),
                "title": table.get("documentTitle", []),
                "columns": table.get("columns", []),
                "rows": table.get("rows", [])[:row_limit]
            }
        case _ as dataset:
            typing.assert_never(dataset)

    return json.dumps(table_content, separators=(",", ":"))


def csv_table_rep(dataset: Dataset, table: dict, row_limit: int | None) -> str:
    csv_output = io.StringIO()
    csv_writer = csv.writer(csv_output, delimiter=',')
    match dataset:
        case Dataset.AITQA:
            table_id = table["id"]
            title = table["title"]
            column_headers = [header[0] + " " + header[1] if len(header) > 1 else header[0] for header in
                              table["column_header"]]
            row_headers = [header[0] + " " + header[1] if len(header) > 1 else header[0] for header in
                           table["row_header"]]
            data = table["data"]
            csv_writer.writerow([table_id] + [title])
            if row_headers:
                csv_writer.writerow([''] + column_headers)
            else:
                csv_writer.writerow(column_headers)
            if row_headers:
                for row_header, row_data in zip(row_headers[:row_limit], data[:row_limit]):
                    csv_writer.writerow([row_header] + row_data)
            else:
                for row_data in data[:row_limit]:
                    csv_writer.writerow(row_data)
        case Dataset.NQTABLES:
            table_id = table["tableId"]
            title = table["documentTitle"]
            columns = [column["text"] for column in table["columns"]]
            rows = [[cell["text"] for cell in row["cells"]] for row in table["rows"][:row_limit]]
            csv_writer.writerow([table_id] + [title])
            csv_writer.writerow(columns)
            csv_writer.writerows(row for row in rows)

    table_content = csv_output.getvalue()
    csv_output.close()
    return table_content


def get_table_title(dataset: Dataset, linearization: Linearization, table: str) -> str:
    match linearization:
        case Linearization.JSON:
            table_dict = json.loads(table)
            match dataset:
                case Dataset.AITQA:
                    return table_dict["title"]
                case Dataset.NQTABLES:
                    return table_dict["documentTitle"]
                case _ as dataset:
                    typing.assert_never(dataset)
        case Linearization.CSV:
            csv_file = io.StringIO(table)
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader, None)
            return first_row[0]
        case _ as linearization:
            typing.assert_never(linearization)


@functools.cache
def get_tables_cached(dataset: Dataset) -> dict[str, dict]:
    table_paths = get_tables_path(dataset)
    tables = {}
    with open(table_paths, "r", encoding="utf-8") as all_tables:
        for line in all_tables:
            table = json.loads(line)
            match dataset:
                case Dataset.AITQA:
                    table_id = table["id"]
                case Dataset.NQTABLES:
                    table_id = table["tableId"]
                case _ as dataset:
                    typing.assert_never(dataset)
            tables[table_id] = table
        return tables


def get_serialized_table_by_id(
        dataset: Dataset,
        preprocess: bool,
        llm_model_name: str,
        token_limit: int | None,
        row_limit: int | None,
        linearization: Linearization,
        target_id: str
) -> str | None:
    tables = get_tables_cached(dataset)
    table = tables[target_id]

    if preprocess:
        table = preprocess_table(dataset, table, llm_model_name, token_limit)
    match linearization:
        case Linearization.JSON:
            return json_table_rep(dataset, table, row_limit)
        case Linearization.CSV:
            return csv_table_rep(dataset, table, row_limit)
        case _ as linearization:
            typing.assert_never(linearization)


def load_table_data(dataset: Dataset, linearization: Linearization, line: str) -> TableData:
    table = json.loads(line)

    match dataset:
        case dataset.AITQA:
            table_id = table["id"]
            table_name = table["title"]
        case dataset.NQTABLES:
            table_id = table["tableId"]
            table_name = table["documentTitle"]
        case _ as dataset:
            typing.assert_never(dataset)

    match linearization:
        case Linearization.JSON:
            table_content = json_table_rep(dataset, table, None)
        case Linearization.CSV:
            table_content = csv_table_rep(dataset, table, None)
        case _ as linearization:
            typing.assert_never(linearization)

    return TableData(table_id, table_name, table_content)


def load_query_data(dataset: Dataset, line: str) -> QueryData:
    query_data = json.loads(line)

    match dataset:
        case Dataset.AITQA:
            return QueryData(
                query_id=query_data["id"],
                query_text=query_data["question"],
                table_id=query_data["table_id"],
                ground_truth=query_data["answers"],
                generated_answer=None
            )
        case Dataset.NQTABLES:
            query_dict = query_data["questions"][0]
            return QueryData(
                query_id=query_dict["id"],
                query_text=query_dict["originalText"],
                table_id=query_data["table"]["tableId"],
                ground_truth=query_dict["answer"]["answerTexts"],
                generated_answer=None
            )
        case _ as dataset:
            typing.assert_never(dataset)


def fill_template(template: str, **args) -> str:
    def replace_variable(match) -> str:
        variable = match.group(1)
        if variable not in args.keys():
            raise AssertionError(f"Missing value for template string variable {variable}!")
        return args[variable]

    return re.sub(r"\{\{([^{}]+)\}\}", replace_variable, template)
