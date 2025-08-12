import abc
import json
import logging
import typing

import attrs
import omegaconf
import tqdm
from hydra.core.config_store import ConfigStore

from lib.llms import execute
from lib.models import Dataset, QueryData, SearchHit, Linearization
from lib.utils import get_serialized_table_by_id

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()


########################################################################################################################
# BASE ZOOM
########################################################################################################################

@attrs.define
class BaseZoomConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseZoom(abc.ABC):
    dataset: Dataset
    linearization: Linearization
    num_zoom: int
    llm_model_name: str
    llm_seed: int

    def prepare(self) -> None:
        pass  # default: do nothing

    @abc.abstractmethod
    def zoom(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        raise NotImplementedError()


########################################################################################################################
# NO ZOOM
########################################################################################################################

@attrs.define
class NoZoomConfig(BaseZoomConfig):
    _target_: str = "lib.zoom.NoZoom"


cs.store(name="no_zoom", node=NoZoomConfig, group="zoom")


@attrs.define
class NoZoom(BaseZoom):

    def zoom(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        return all_search_hits


########################################################################################################################
# ROW BASED ZOOM
########################################################################################################################

class NextFocusSize(typing.TypedDict):
    num_tables: int
    num_rows: int


@attrs.define
class RowBasedZoomConfig(BaseZoomConfig):
    _target_: str = "lib.zoom.RowBasedZoom"
    next_focus_sizes: list[NextFocusSize] = [
        {
            "num_tables": 10,  # reduce to this many tables
            "num_rows": 1
        },
        {
            "num_tables": 3,  # reduce to this many tables
            "num_rows": 5
        }
    ]
    temperature: float = 0.0
    max_response_tokens: int = 1000
    system_msg: str = """You are a data specialist who selects relevant tables that may help answer a given question.
Select the specified number of tables that you believe are most likely to contain the answer to the given question. Order them from most likely to least likely.
Provide the table_id strings of these tables formatted as a JSON list: ["table_id_1", "table_id_2", "..."]
Provide only the JSON without any explanations."""
    prompt_template: str = """Provide the table_id strings of the top {next_focus_size} that most likely contain the answer to the following question:

{query}

{tables}
"""


cs.store(name="rowbased_zoom", node=RowBasedZoomConfig, group="zoom")


@attrs.define
class RowBasedZoom(BaseZoom):
    next_focus_sizes: list[NextFocusSize]
    temperature: float
    max_response_tokens: int
    system_msg: str
    prompt_template: str

    def zoom(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        focus_k = self.num_zoom
        for next_focus_size in self.next_focus_sizes:
            logger.info(f"reduce number of tables from {focus_k} to {next_focus_size['num_tables']}")
            assert next_focus_size["num_tables"] < focus_k, \
                f"cannot reduce number of tables from {focus_k} to {next_focus_size['num_tables']}"

            all_requests = []
            for query, search_hits in zip(tqdm.tqdm(all_queries, "prepare zoom requests"), all_search_hits):
                all_requests.append(self.make_request(query, search_hits[:focus_k], next_focus_size))

            all_responses = execute(all_requests, auto_transform="openai", budget=None)

            new_all_search_hits = []
            for query, search_hits, response in zip(tqdm.tqdm(all_queries, "parse zoom responses"), all_search_hits,
                                                    all_responses):
                selection = self.parse_response(search_hits[:focus_k], response)
                reordered_search_hits = self.reorder_search_hits(
                    search_hits[:focus_k],
                    selection,
                    next_focus_size["num_tables"]
                )
                new_all_search_hits.append(reordered_search_hits + search_hits[focus_k:])
            all_search_hits = new_all_search_hits

            focus_k = next_focus_size["num_tables"]

        return all_search_hits

    def make_request(self, query: QueryData, search_hits: list[SearchHit], next_focus_size: NextFocusSize) -> dict:
        serialized_tables_parts = []
        for search_hit in search_hits:
            table_content = get_serialized_table_by_id(
                dataset=self.dataset,
                preprocess=False,  # for zoom-reduced, set this to True
                llm_model_name=self.llm_model_name,
                token_limit=None,  # for zoom-reduced, set this to 10
                row_limit=next_focus_size["num_rows"],
                linearization=self.linearization,
                target_id=search_hit.table_id
            )
            serialized_tables_parts.append(f"table_id: {search_hit.table_id}\n\n{table_content}")

        prompt_values = {
            "query": query.query_text,
            "tables": "\n\n\n".join(serialized_tables_parts),
            "next_focus_size": str(next_focus_size)
        }
        return {
            "model": self.llm_model_name,
            "messages": [
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": self.prompt_template.format(**prompt_values)}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_response_tokens,
            "seed": self.llm_seed
        }

    @staticmethod
    def parse_response(search_hits: list[SearchHit], response: dict) -> list[bool]:
        selection = [False] * len(search_hits)
        try:
            text = response["choices"][0]["message"]["content"]
            if text.startswith("```json"):
                text = text[len("```json"):]
            if text.endswith("```"):
                text = text[:-len("```")]
            text = text.strip()
            selected_ids = json.loads(text)
            assert isinstance(selected_ids, list)
            selected_ids = set(selected_ids)
            for ix, search_hit in enumerate(search_hits):
                if search_hit.table_id in selected_ids:
                    selection[ix] = True
        except Exception as e:
            logger.warning(f"failed to parse response, won't reorder before focusing: {e}, {response}")
        return selection

    @staticmethod
    def reorder_search_hits(search_hits: list[SearchHit], selection: list[bool], num_tables: int) -> list[SearchHit]:
        """Place selected search hits in front of not selected search hits, otherwise keeps original order."""
        num_selected = sum(selection)
        if num_selected != num_tables:
            logger.warning(f"LLM should have selected {num_tables} tables, but selected {num_selected} tables")
        selected_search_hits = [hit for hit, selected in zip(search_hits, selection) if selected]
        not_selected_search_hits = [hit for hit, selected in zip(search_hits, selection) if not selected]
        return selected_search_hits + not_selected_search_hits
