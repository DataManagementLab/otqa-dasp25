import json
import logging
import os
import re
from pathlib import Path
from typing import Union

import attrs
import cattrs
import hydra
from hydra.core.config_store import ConfigStore
from tqdm import tqdm

from lib.data import get_data_path
from lib.models import Dataset, EvalData

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    llm_model_name: str = "gpt-4o-mini-2024-07-18"
    partitions: list[str] = ["train", "dev", "test"]


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    logger.info(f"Starting splitting the queries in the NQ-Tables dataset.")

    nq_dir = get_data_path() / Dataset.NQTABLES.value
    nq_dir.mkdir(parents=True, exist_ok=True)

    for partition in cfg.partitions:
        logger.info(f"Preparing for splitting the {partition} partition.")

        runs_dir = nq_dir / "experiments" / partition / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # define file paths for the easy and hard queries for the given partition
        classified_dir = nq_dir / "classified" / partition
        os.makedirs(classified_dir, exist_ok=True)
        nq_easy_path = classified_dir / "easy.jsonl"
        nq_hard_path = classified_dir / "hard.jsonl"

        pattern = (
            f"retriever=lib.retrieval.NoRetriever "
            f"reranker=lib.reranking.NoReranker "
            f"zoom=lib.zoom.NoZoom "
            f"generator=lib.generation.LLMGenerator "
            f"model={cfg.llm_model_name} "
            f"num_context=*.json"
        )
        matches = list(runs_dir.glob(pattern))

        # continue with next partition if there was no closed-book QA performed on the current partition
        if len(matches) == 0:
            continue

        # read the experiment data from the experiment data file
        converter = cattrs.Converter()
        converter.register_structure_hook(Union[str, dict[str, list]], structure_union_str_dict)
        with open(matches[0], 'r') as file:
            eval_data = converter.structure(json.load(file), EvalData)

        preds = eval_data.qa_eval_data.predictions
        refs = eval_data.qa_eval_data.references

        with open(nq_easy_path, 'w') as nq_easy_file, open(nq_hard_path, 'w') as nq_hard_file:
            # for every reference-prediction pair, find the corresponding question and write it to the nq easy file if
            # the question was answered correctly by GPT without context and to nq hard file otherwise
            for prediction, reference in tqdm(zip(preds, refs), desc=f"Splitting {partition}", unit="items",
                                              dynamic_ncols=True, total=len(preds)):

                # to load the query we need the id AND reference answers -> id is NOT unique
                answer_texts = [re.sub(r"^'(.*)'$", r'"\1"', ref) for ref in reference["answers"]["text"]]
                query = load_nq_query_by_id(nq_dir, partition, prediction["id"], answer_texts)
                if query is None:
                    continue

                if has_relaxed_match(prediction["prediction_text"], answer_texts):
                    nq_easy_file.write(f"{json.dumps(query)}\n")
                else:
                    nq_hard_file.write(f"{json.dumps(query)}\n")

    logger.info(f"Completed splitting the queries of the NQ-Tables dataset.")


def structure_union_str_dict(obj: Union[str, dict[str, list]], cl: type) -> Union[str, dict[str, list]]:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot structure {obj} as {cl}")


def load_nq_query_by_id(nq_dir: Path, partition: str, target_id: str, answer_texts: list[str]) -> Union[None, dict]:
    """Loads the query with the given query id."""
    subset_path = nq_dir / "interactions" / f"{partition}.jsonl"
    with open(subset_path, 'r') as file:
        for line in file:
            query = json.loads(line)
            query_answers = query["questions"][0]["answer"]["answerTexts"]
            if query["questions"][0]["id"] == target_id and query_answers == answer_texts:
                return query
    return None


def has_relaxed_match(
        prediction: str,
        references: list[str]
) -> bool:
    """Verifies whether the prediction contains the reference as substring."""
    """if bool(re.search(re.escape(references[0].lower()), prediction.lower())):
        logger.info(f"Old would be matched! Prediction: {prediction} References: {references}")
        if all(re.search(re.escape(ref.lower()), prediction_lower) for ref in references):
            logger.info("Full match!")
        else:
            logger.info("No match!")
        return all(re.search(re.escape(ref.lower()), prediction_lower) for ref in references)
    return False"""
    prediction_lower = prediction.lower()
    return all(re.search(re.escape(ref.lower()), prediction_lower) for ref in references)


if __name__ == '__main__':
    main()
