import abc
import json
import logging

import attrs
import numpy as np
import omegaconf
import torch
from hydra.core.config_store import ConfigStore
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from lib.llms import execute
from lib.models import Dataset, SearchHit, QueryData, FilteredResponses, HitResponsePair, Linearization
from lib.utils import get_serialized_table_by_id, get_table_title, fill_template

logger = logging.getLogger(__name__)
cs = ConfigStore.instance()


########################################################################################################################
# BASE RERANKER
########################################################################################################################

@attrs.define
class BaseRerankerConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseReranker(abc.ABC):
    dataset: Dataset
    linearization: Linearization
    num_rerank: int
    llm_model_name: str
    llm_seed: int

    def prepare(self) -> None:
        pass  # default: do nothing

    @abc.abstractmethod
    def rerank(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        raise NotImplementedError()


########################################################################################################################
# NO RERANKER
########################################################################################################################

@attrs.define
class NoRerankerConfig(BaseRerankerConfig):
    _target_: str = "lib.reranking.NoReranker"


cs.store(name="no_reranker", node=NoRerankerConfig, group="reranker")


@attrs.define
class NoReranker(BaseReranker):

    def rerank(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        return all_search_hits


########################################################################################################################
# QUERY-BASED RERANKER
########################################################################################################################

@attrs.define
class QueryBasedRerankerConfig(BaseRerankerConfig):
    _target_: str = "lib.reranking.QueryBasedReranker"
    embedder_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    temperature: float = 0.0
    max_response_tokens: int = 400
    system_msg: str = """You are a chatbot that generates questions from tables. Provide only the generated questions in the following json format without any explanation: {"<table_id>": ["<question1_text>", "<question2_text>", ..., "<question10_text>"]}! IMPORTANT: Do not change the provided table id!"""
    prompt_templates: dict[Dataset, str] = {
        Dataset.AITQA: """Generate 10 questions from the table at the end of the prompt. Include the airline name in the questions: {{airline_name}}. Use the provided table id exactly as shown here: "{{table_id}}". Table: {{table}}""",
        Dataset.NQTABLES: """Generate 10 questions from the table at the end of the prompt. Use the provided table id exactly as shown here: "{{table_id}}". Table: {{table}}"""
    }


cs.store(name="query_based_reranker", node=QueryBasedRerankerConfig, group="reranker")


@attrs.define
class QueryBasedReranker(BaseReranker):
    embedder_model_name: str
    temperature: float
    max_response_tokens: int
    system_msg: str
    prompt_templates: dict[Dataset, str]
    embedder: SentenceTransformer = attrs.field(init=False)
    embeddings_cache: dict[str, torch.Tensor] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self.embeddings_cache = {}

    def prepare(self) -> None:
        logger.info(f"start loading embedder model from {self.embedder_model_name}")
        self.embedder = SentenceTransformer(self.embedder_model_name)
        logger.info(f"loaded embedder model")

    def rerank(self, all_queries: list[QueryData], all_search_hits: list[list[SearchHit]]) -> list[list[SearchHit]]:
        all_results = []
        for query, search_hits in zip(all_queries, all_search_hits):
            top_n_hits, rest_hits = search_hits[:self.num_rerank], search_hits[self.num_rerank:]

            logger.debug(f"start re-ranking the top {self.num_rerank} retrieved tables for query {query.query_id}: "
                         f"{[search_hit.table_id for search_hit in top_n_hits]}")

            # construct a list of requests to generate 10 queries from each of the top n tables
            requests = []
            for search_hit in top_n_hits:
                table_content = get_serialized_table_by_id(
                    dataset=self.dataset,
                    preprocess=False,
                    llm_model_name=self.llm_model_name,
                    token_limit=None,
                    row_limit=None,
                    linearization=self.linearization,
                    target_id=search_hit.table_id
                )
                request = {
                    "model": self.llm_model_name,
                    "messages": [
                        {"role": "system", "content": self.system_msg},
                        {"role": "user", "content": self.format_prompt(
                            template=self.prompt_templates.get(self.dataset),
                            airline_name=get_table_title(
                                self.dataset,
                                self.linearization,
                                table_content
                            ) if self.dataset == Dataset.AITQA else None,
                            table_id=search_hit.table_id,
                            table=table_content
                        )},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_response_tokens,
                    "seed": self.llm_seed
                }
                requests.append(request)

            responses = execute(requests, budget=None, auto_transform="openai")

            # responses must be filtered, since some may have failed
            filtered_responses = self.filter_openai_responses(top_n_hits, responses)

            # add all tables with a failed request to rest_hits (they will not be re-ranked!) and remove them from top_n_hits
            if len(filtered_responses.hits_with_failed_request) > 0:
                top_n_hits = [elem.search_hit for elem in filtered_responses.hit_response_pairs]
                rest_hits = filtered_responses.hits_with_failed_request + rest_hits

            # in case all LLM requests failed, the old ranking is returned
            if not top_n_hits:
                logger.warning(
                    f"re-ranking the top {self.num_rerank} retrieved tables for query {query.query_id} failed")
                all_results.append(search_hits)
                continue

            # for each search hit with a successful LLM request: calculate the max similarity score of the generated queries
            # to the original query
            re_ranked_tables = [
                self.calculate_table_max_similarity(query, hit_response_pair.search_hit, hit_response_pair.response)
                for hit_response_pair in filtered_responses.hit_response_pairs
            ]

            # re-ranking was performed using cosine similarity with scores between 0 and 1
            # BM25 scores are usually higher, the scores of the tables that were not re-ranked have to be normalized
            # re_ranked_tables = [(table.table_id, score) for table, score in re_ranked_tables]
            if rest_hits:
                combined = self.normalize_scores(re_ranked_tables, rest_hits)
            else:
                combined = list(sorted(re_ranked_tables, key=lambda hit: hit.score, reverse=True))
            logger.debug(
                f"completed re-ranking for query {query.query_id} with the new ranking: {[(search_hit.table_id, search_hit.score) for search_hit in combined]}"
            )

            # return ranking of tables and their reranking scores (internal id should be removed)
            all_results.append(combined)
        return all_results

    def format_prompt(
            self,
            template: str,
            airline_name: str | None,
            table_id: str,
            table: str
    ) -> str:
        """Formats the prompt with the given values. Parameter airline_name is optional (only needed for dataset AIT-QA."""
        values = {
            "table_id": table_id,
            "table": table,
        }

        if airline_name is not None:
            values["airline_name"] = airline_name

        return fill_template(template, **values)

    def filter_openai_responses(
            self,
            top_n_hits: list[SearchHit],
            responses: list[dict]
    ) -> FilteredResponses:
        """Filter the LLM responses so that tables that were expected to be re-ranked, but have a failed LLM request are
        removed from the list of tables that actually will be re-ranked."""

        # step 1: parse responses (check if LLM response is parseable and filter out failed requests)
        hit_response_pairs, hits_with_failed_request = [], []
        parsed_table_ids = set()

        for idx, response in enumerate(responses):
            choices = response.get('choices', [{}])
            if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                content = choices[0]['message']['content']
                try:
                    # parse generated questions
                    if content.startswith("```json"):
                        content = content[len("```json"):]
                    if content.endswith("```"):
                        content = content[:-len("```")]
                    content = content.strip()
                    generated_questions = json.loads(content)
                    table_id = list(generated_questions.keys())[0]

                    # step 2: match the response to the table and store valid responses
                    for search_hit in top_n_hits:
                        if search_hit.table_id == table_id:
                            hit_response_pairs.append(HitResponsePair(search_hit, generated_questions))
                            parsed_table_ids.add(table_id)
                            logger.debug(f"Generated questions: {generated_questions}")
                except json.decoder.JSONDecodeError as e:
                    logger.info(f"JSONDecodeError at index {idx}: {e}, Content: {content}")
            else:
                logger.info(f"Missing or invalid 'choices' structure at index {idx}: {response}")

        # step 4: collect tables that had an invalid LLM response
        hits_with_failed_request = [
            table for table in top_n_hits if table.table_id not in parsed_table_ids
        ]

        if hits_with_failed_request:
            logger.warning(f"LLM requests have failed for: {hits_with_failed_request}")

        return FilteredResponses(hit_response_pairs, hits_with_failed_request)

    def calculate_table_max_similarity(
            self,
            query: QueryData,
            search_hit: SearchHit,
            response: dict
    ) -> SearchHit:
        """Calculates the maximum similarity score the generated queries can achieve to the original query."""
        # load all questions generated for the table
        generated_queries = response[search_hit.table_id]

        # determine the maximum similarity score that all generated queries can achieve to the original question
        similarity_scores = self.calculate_similarities(query, generated_queries)
        max_similarity_score = max(similarity_scores, default=0.0)

        logger.debug(f"table {search_hit.table_id} has max similarity score: {max_similarity_score}")

        # return the updated search hit
        return SearchHit(query_id=query.query_id, table_id=search_hit.table_id, score=max_similarity_score)

    def calculate_similarities(
            self,
            query: QueryData,
            generated_queries: list[str]
    ) -> list[float]:
        """Calculates the cosine similarities of the LLM-generated queries to the original query."""
        # step 1: embed the reference query and all generated queries
        reference_embedding = self.cached_embedder(query.query_text)

        generated_queries_embeddings = [self.cached_embedder(gen_query) for gen_query in generated_queries]

        # step 2: calculate the cosine distances of the generated queries to the reference query
        distances = cosine_distances(generated_queries_embeddings,
                                     reference_embedding.reshape((1, *reference_embedding.shape)))
        similarities = np.ones_like(distances) - distances

        return [float(score) for score in similarities.reshape((similarities.shape[0],))]

    def normalize_scores(
            self,
            re_ranked_hits: list[SearchHit],
            rest_hits: list[SearchHit]
    ) -> list[SearchHit]:
        """Combine the re-ranked search hits with the rest of the search hits and normalize their scores so that the
        lowest re-ranked score is the upper bound for the non-re-ranked hits."""

        # re_ranked_scores contains updated cosine similarity scores from the re-ranked search hits
        re_ranked_scores = [hit.score for hit in re_ranked_hits]
        # rest_scores contains the old ranking scores of the non-re-ranked search hits
        rest_scores = [hit.score for hit in rest_hits]

        # since the old ranking scores might be >1, the scores of the non-re-ranked search hits must be normalized with
        # the lowes re-ranked score as upper bound for the normalized scores
        min_re_ranked_score = min(re_ranked_scores)
        max_rest_score = max(rest_scores)
        epsilon = 0.01

        normalized_rest_list = [SearchHit(
            query_id=hit.query_id,
            table_id=hit.table_id,
            score=float(hit.score * (min_re_ranked_score - epsilon) / max_rest_score)
        ) for hit in rest_hits]

        # combine the list of re_ranked search hits and non-re-ranked search hits with their normalized scores
        result_scores = sorted(re_ranked_hits + normalized_rest_list, key=lambda x: x.score, reverse=True)
        return result_scores

    def cached_embedder(self, s: str) -> torch.Tensor:
        if s not in self.embeddings_cache.keys():
            self.embeddings_cache[s] = self.embedder.encode(s, show_progress_bar=False)
        return self.embeddings_cache[s]
