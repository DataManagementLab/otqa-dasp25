import json
import logging
import pathlib
import random
import re
import string
import typing
from enum import Enum

import attrs
import cattrs
import evaluate
import hydra
import numpy as np
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from lib import llms
from lib.data import get_data_path, get_queries_path
from lib.generation import NoGenerator
from lib.llms import cost
from lib.models import Dataset, QAMode, RequestData, EvalData, RetrievalEvalData, QAEvalData, Linearization, QAResults, \
    SearchHit, Template
from lib.reranking import NoReranker
from lib.retrieval import NoRetriever, OracleRetriever
from lib.utils import load_query_data, get_serialized_table_by_id
from lib.zoom import NoZoom

logger = logging.getLogger(__name__)
_, _, _, _ = NoRetriever, NoZoom, NoReranker, NoGenerator  # we need these imports


@attrs.define
class Config:
    defaults: list = [
        {"retriever": MISSING},
        {"reranker": MISSING},
        {"zoom": MISSING},
        {"generator": MISSING},
        "_self_"
    ]
    exp: str = MISSING
    config_name: str = MISSING
    random_seed: int = 230032298

    dataset: Dataset = MISSING
    partition: str = MISSING
    limit_queries: int | None = None

    linearization: Linearization = MISSING
    template: Template = MISSING
    num_retrieve: int = 100
    num_zoom: int = 100
    num_rerank: int = 10
    num_context: int = MISSING

    pre_processing: bool = False
    token_limit: int | None = None
    row_limit: int | None = None

    llm_model_name: str = MISSING
    llm_seed: int = 732172265


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    random.seed(cfg.random_seed)

    # prepare queries
    queries_path = get_queries_path(dataset=cfg.dataset, subset=cfg.partition)
    with open(queries_path, "r", encoding="utf-8") as file:
        all_queries = [load_query_data(cfg.dataset, line) for line in tqdm.tqdm(file, desc="loading queries")]

    if cfg.limit_queries is not None:
        logger.info("shuffling queries before limiting them")
        random.shuffle(all_queries)

    if cfg.limit_queries is not None and cfg.limit_queries > len(all_queries):
        logger.warning(f"{cfg.partition} partition of {cfg.dataset.name} contains only {len(all_queries)} queries, "
                       f"which is lower than cfg.limit_queries={cfg.limit_queries}")
        cfg.limit_queries = len(all_queries)
    all_queries = all_queries[:cfg.limit_queries]

    retriever = instantiate(
        cfg.retriever,
        dataset=cfg.dataset,
        linearization=cfg.linearization,
        num_retrieve=cfg.num_retrieve
    )

    if isinstance(retriever, NoRetriever):
        qa_mode = QAMode.CLOSED
    elif isinstance(retriever, OracleRetriever):
        qa_mode = QAMode.ORACLE
    else:
        qa_mode = QAMode.OPEN

    zoom = instantiate(
        cfg.zoom,
        dataset=cfg.dataset,
        linearization=cfg.linearization,
        num_zoom=cfg.num_zoom,
        llm_model_name=cfg.llm_model_name,
        llm_seed=cfg.llm_seed
    )
    reranker = instantiate(
        cfg.reranker,
        dataset=cfg.dataset,
        linearization=cfg.linearization,
        num_rerank=cfg.num_rerank,
        llm_model_name=cfg.llm_model_name,
        llm_seed=cfg.llm_seed
    )
    generator = instantiate(
        cfg.generator,
        dataset=cfg.dataset,
        qa_mode=qa_mode,
        llm_model_name=cfg.llm_model_name,
        llm_seed=cfg.llm_seed,
        template=cfg.template
    )

    # prepare the objects that will store the evaluation data
    eval_data = EvalData(
        cutoff_context_ctr=0,
        processed_queries_ctr=0,
        processed_queries=None,
        retrieval_eval_data=RetrievalEvalData(run={}, qrels={}),
        qa_eval_data=QAEvalData(predictions=[], references=[]),
        retrieval_results=None,
        qa_results=None,
        total_llm_cost=0,
        config=serialize_config(OmegaConf.to_container(cfg, resolve=True))
    )

    retriever.prepare()
    zoom.prepare()
    reranker.prepare()

    all_search_hits = retriever.retrieve(all_queries)
    all_search_hits = normalize_scores(all_search_hits)
    all_search_hits = zoom.zoom(all_queries, all_search_hits)
    all_search_hits = normalize_scores(all_search_hits)
    all_search_hits = reranker.rerank(all_queries, all_search_hits)
    all_search_hits = normalize_scores(all_search_hits)
    for query, search_hits in zip(all_queries, all_search_hits):
        eval_data.retrieval_eval_data.run[query.query_id] = {
            hit.table_id: hit.score for hit in search_hits
        }
        eval_data.retrieval_eval_data.qrels[query.query_id] = {query.table_id: 1}

    logger.info("execute question answering")
    # 1. STEP: gather the data needed for building an LLM request including query id, query text, question answering
    # mode and optionally (depending on the qa mode) the context tables
    all_requests_data = []
    for query_ix, (query, search_hits) in enumerate(
            zip(tqdm.tqdm(all_queries, "prepare question answering"), all_search_hits)
    ):
        context = None

        match qa_mode:
            case QAMode.OPEN:
                context = [
                    get_serialized_table_by_id(
                        cfg.dataset,
                        cfg.pre_processing,
                        cfg.llm_model_name,
                        cfg.token_limit,
                        cfg.row_limit,
                        cfg.linearization,
                        hit.table_id
                    )
                    for hit in search_hits[:cfg.num_context]
                ]
            case QAMode.ORACLE:
                context = [get_serialized_table_by_id(
                    cfg.dataset,
                    cfg.pre_processing,
                    cfg.llm_model_name,
                    cfg.token_limit,
                    cfg.row_limit,
                    cfg.linearization,
                    search_hits[0].table_id
                )]
            case QAMode.CLOSED:
                context = None
            case _ as qa_mode:
                typing.assert_never(qa_mode)

        request = RequestData(
            query_id=query.query_id,
            query_text=query.query_text,
            qa_mode=qa_mode,
            context=context
        )
        all_requests_data.append(request)

    # 2. STEP: question answering
    logger.info("generate answers")
    all_generated_answers = generator.generate(all_requests_data)
    assert len(all_generated_answers) == len(all_queries), "some requests failed"

    references, predictions = [], []
    for generated_answer, query in zip(all_generated_answers, all_queries):
        query.generated_answer = generated_answer
        # answer_start is required by the evaluation metric SQuAD (but not used), we thus set it to an
        # invalid value for all generated answers
        references.append({
            "id": query.query_id,
            "answers": {
                "text": query.ground_truth,
                "answer_start": [-1] * len(query.ground_truth),
            }
        })
        predictions.append({
            "id": query.query_id,
            "prediction_text": generated_answer
        })

    eval_data.qa_eval_data.predictions = predictions
    eval_data.qa_eval_data.references = references

    # add the data of all processed questions to the eval data and write it to the file
    eval_data.cutoff_context_ctr = generator.cutoff_context_ctr
    eval_data.processed_queries_ctr = len(all_queries)
    eval_data.processed_queries = all_queries

    qrels = eval_data.retrieval_eval_data.qrels
    run = eval_data.retrieval_eval_data.run
    k_values = [1, 3, 5, 10, 100]
    top_k_acc = top_k_accuracy(qrels, run, k_values)
    eval_data.retrieval_results = top_k_acc
    logger.info(f"retrieval results: {json.dumps(eval_data.retrieval_results)}")

    # evaluate the question answering results using the metrics exact match, f1 score and relaxed match
    predictions = eval_data.qa_eval_data.predictions
    references = eval_data.qa_eval_data.references
    squad_results = evaluate_squad(predictions, references)
    relaxed_match_results = relaxed_match(predictions, references)
    qa_results = QAResults(
        exact_match=squad_results["exact_match"] / 100,
        relaxed_match=relaxed_match_results,
        f1_score=squad_results["f1"] / 100
    )
    eval_data.qa_results = qa_results
    logger.info(f"qa results: {eval_data.qa_results}")

    responses = []
    for response in llms.RESPONSES:  # filter out failed responses
        if "model" in response.keys():
            responses.append(response)
    eval_data.total_llm_cost = sum(
        cost(responses, ignore_token_caching=True, auto_transform="openai")) / eval_data.processed_queries_ctr

    results_path = get_data_path() / cfg.dataset.value / "experiments" / cfg.partition / "runs" / f"{cfg.config_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(cattrs.unstructure(eval_data), file, indent=2, ensure_ascii=False)

    logger.info(f"experiment saved to {results_path}")


def serialize_config(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_config(v) for v in value]
    if isinstance(value, dict):
        return {serialize_config(k): serialize_config(v) for k, v in value.items()}
    return value


def top_k_accuracy(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int]
) -> dict[str, float]:
    """Computes top-k accuracy for the given query relevance and results. The code of this function was copied from the
    BEIR benchmark (beir/retrieval/custom_metrics.py).

    Args:
        qrels: A dictionary containing the query relevance scores.
        results: A dictionary containing the retrieval results.
        k_values: A list of k values for which to compute the top-k accuracy.

    Returns:
        A dictionary containing the top k accuracy results for each k value.
    """
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in
                              sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]

    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)

    return top_k_acc


def evaluate_squad(predictions: dict, references: dict) -> dict:
    """Evaluates the results of the question answering using the Exact Match accuracy and F1 score metrics from SQuAD.

    Args:
        predictions: A dictionary containing the LLM-generated predictions.
        references: A dictionary containing the ground truths.

    Returns:
        A dictionary containing the results of the Exact Match accuracy and F1 score metrics.
    """
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    return results


def relaxed_match(predictions: dict, references: dict) -> float:
    """Evaluates the predictions using a less strict approach of Exact Match accuracy. It ignores punctuation (except
    for commas and periods) and casing and verifies whether the reference is a substring of the prediction  allowing
    generated answers to be incorporated into a larger string like a sentence without being penalized for it.

    Args:
        predictions: A dictionary containing the LLM-generated predictions.
        references: A dictionary containing the reference answers (ground truth).

    Returns:
        The mean Relaxed Match accuracy score across all predictions.
    """
    # create a translation table to remove punctuation characters (except for commas and periods)
    translator = str.maketrans('', '', string.punctuation.replace(',', '').replace('.', ''))

    score_list = []

    # iterate over all pairs of ground truths and predictions
    for pred, ref in zip(predictions, references):
        # ignore casing of the ground truth and generated answer
        generated_answer = pred["prediction_text"].translate(translator)
        ground_truth = ref["answers"]["text"][0].translate(translator)

        # check whether the generated answer contains a substring that matches the ground truth
        match = re.search(re.escape(ground_truth), generated_answer, re.IGNORECASE)
        score_list.append(True if match else generated_answer.lower() == ground_truth.lower())

    return float(np.mean(score_list))


def normalize_scores(all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
    for search_hits in all_search_hits:
        for ix, search_hit in enumerate(search_hits):
            search_hit.score = 1 / (ix + 1)
    return all_search_hits


if __name__ == "__main__":
    main()
