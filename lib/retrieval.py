import abc
import logging
import pickle

import attrs
import nltk
import omegaconf
import rank_bm25
import torch
import tqdm
from hydra.core.config_store import ConfigStore
from nltk import RegexpTokenizer, PorterStemmer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

from lib.data import get_tables_path
from lib.models import Dataset, TableData, QueryData, SearchHit, Linearization
from lib.utils import load_table_data

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

logger = logging.getLogger(__name__)
cs = ConfigStore.instance()


########################################################################################################################
# BASE RETRIEVER
########################################################################################################################

@attrs.define
class BaseRetrieverConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseRetriever(abc.ABC):
    dataset: Dataset
    linearization: Linearization
    num_retrieve: int

    def prepare(self) -> None:
        pass  # default: do nothing

    @abc.abstractmethod
    def retrieve(self, all_queries: list[QueryData]) -> list[list[SearchHit]]:
        raise NotImplementedError()


########################################################################################################################
# NO RETRIEVER
########################################################################################################################

@attrs.define
class NoRetrieverConfig(BaseRetrieverConfig):
    _target_: str = "lib.retrieval.NoRetriever"


cs.store(name="no_retriever", node=NoRetrieverConfig, group="retriever")


@attrs.define
class NoRetriever(BaseRetriever):

    def retrieve(self, all_queries: list[QueryData]) -> list[list[SearchHit]]:
        return [[] for _ in all_queries]


########################################################################################################################
# ORACLE RETRIEVER
########################################################################################################################

@attrs.define
class OracleRetrieverConfig(BaseRetrieverConfig):
    _target_: str = "lib.retrieval.OracleRetriever"


cs.store(name="oracle_retriever", node=OracleRetrieverConfig, group="retriever")


@attrs.define
class OracleRetriever(BaseRetriever):

    def retrieve(self, all_queries: list[QueryData]) -> list[list[SearchHit]]:
        all_results = []
        for query in tqdm.tqdm(all_queries, "OracleRetriever"):
            all_results.append([SearchHit(query_id=query.query_id, table_id=query.table_id, score=1)])
        return all_results


########################################################################################################################
# BM25 RETRIEVER
########################################################################################################################

@attrs.define
class BM25RetrieverConfig(BaseRetrieverConfig):
    _target_: str = "lib.retrieval.BM25Retriever"
    metadata_separate: bool = True


cs.store(name="bm25_retriever", node=BM25RetrieverConfig, group="retriever")


@attrs.define
class BM25Retriever(BaseRetriever):
    metadata_separate: bool
    tables: list[TableData] = attrs.field(init=False)
    corpus: rank_bm25.BM25 = attrs.field(init=False)
    corpus_names: rank_bm25.BM25 = attrs.field(init=False)

    def prepare(self) -> None:
        tables_path = get_tables_path(self.dataset)
        logger.info(f"start loading tables from {tables_path}")
        self.tables = []
        with open(tables_path, "r") as file:
            for line in file:
                self.tables.append(load_table_data(self.dataset, self.linearization, line))
        logger.info(f"loaded {len(self.tables)} tables")

        table_index_path = tables_path.parent / f"{tables_path.name[:tables_path.name.rindex('.')]}_table_index_{self.linearization}.pickle"
        table_names_index_path = tables_path.parent / f"{tables_path.name[:tables_path.name.rindex('.')]}_table_names_index_{self.linearization}.pickle"
        if not (table_names_index_path.is_file() and table_index_path.is_file()):
            logger.info(f"start preparing BM25 index")
            # tokenize all tables and construct a corpus
            tokenizer = RegexpTokenizer(r"\w+")
            stemmer = PorterStemmer()
            stop_words = stopwords.words("english")
            tokenized_names = []
            tokenized_tables = []
            for table in tqdm.tqdm(self.tables, "tokenize tables and table names"):
                tokenized_table = tokenizer.tokenize(table.table_content)
                tokenized_table = [stemmer.stem(token) for token in tokenized_table if token not in stop_words]
                tokenized_tables.append(tokenized_table)
                tokenized_name = tokenizer.tokenize(table.table_name)
                tokenized_name = [stemmer.stem(token) for token in tokenized_name if token not in stop_words]
                tokenized_names.append(tokenized_name)
            logger.info("start creating corpus_names index")
            self.corpus_names = rank_bm25.BM25Okapi(corpus=tokenized_names, k1=0.9, b=0.4, epsilon=0.01)
            logger.info("created corpus_names index")
            logger.info("start creating corpus index")
            self.corpus = rank_bm25.BM25Okapi(corpus=tokenized_tables, k1=0.9, b=0.4, epsilon=0.01)
            logger.info("created corpus index")
            with open(table_names_index_path, "wb") as file:
                pickle.dump(self.corpus_names, file)
            with open(table_index_path, "wb") as file:
                pickle.dump(self.corpus, file)
            logger.info(f"prepared and saved BM25 index")
        else:
            logger.info(f"start loading table names index from {table_names_index_path}")
            with open(table_names_index_path, "rb") as file:
                self.corpus_names = pickle.load(file)
            logger.info("loaded table names index")
            logger.info(f"start loading table index from {table_index_path}")
            with open(table_index_path, "rb") as file:
                self.corpus = pickle.load(file)
            logger.info("loaded table index")

    def retrieve(self, all_queries: list[QueryData]) -> list[list[SearchHit]]:
        all_results = []
        tokenizer = RegexpTokenizer(r"\w+")
        stemmer = PorterStemmer()
        stop_words = stopwords.words("english")
        for query in tqdm.tqdm(all_queries, "BM25Retriever"):
            tokenized_query = tokenizer.tokenize(query.query_text)
            tokenized_query = [stemmer.stem(token) for token in tokenized_query if token not in stop_words]
            scores = [float(score) for score in self.corpus.get_scores(tokenized_query)]
            if self.metadata_separate:
                scores_names = [float(score) for score in self.corpus_names.get_scores(tokenized_query)]
                scores = [(score + score_name) / 2 for score, score_name in zip(scores, scores_names)]
            results = [
                SearchHit(query_id=query.query_id, table_id=table.table_id, score=score)
                for table, score in zip(self.tables, scores)
            ]
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:self.num_retrieve]
            all_results.append(results)
        return all_results


########################################################################################################################
# SBERT RETRIEVER
########################################################################################################################

@attrs.define
class SBERTRetrieverConfig(BaseRetrieverConfig):
    _target_: str = "lib.retrieval.SBERTRetriever"
    embedder_model_name: str = "all-MiniLM-L6-v2"


cs.store(name="sbert_retriever", node=SBERTRetrieverConfig, group="retriever")


@attrs.define
class SBERTRetriever(BaseRetriever):
    embedder_model_name: str
    embedder: SentenceTransformer = attrs.field(init=False)
    tables: list[TableData] = attrs.field(init=False)
    table_embeddings: torch.Tensor = attrs.field(init=False)

    def prepare(self) -> None:
        logger.info(f"start loading embedder model from {self.embedder_model_name}")
        self.embedder = SentenceTransformer(self.embedder_model_name)
        logger.info("loaded embedder model")

        tables_path = get_tables_path(self.dataset)
        logger.info(f"start loading tables from {tables_path}")
        self.tables = []
        with open(tables_path, "r") as file:
            for line in file:
                table_data = load_table_data(self.dataset, self.linearization, line)
                self.tables.append(table_data)
        logger.info(f"loaded {len(self.tables)} tables")

        embeddings_path = tables_path.parent / f"{tables_path.name[:tables_path.name.rindex('.')]}_embeddings_{self.linearization}.bin"
        if not embeddings_path.is_file():
            logger.info(f"start computing embeddings for {len(self.tables)} tables")
            self.table_embeddings = self.embedder.encode([table.table_content for table in self.tables],
                                                         convert_to_tensor=True)
            torch.save(self.table_embeddings, embeddings_path)
            logger.info(f"computed and saved embeddings")
        else:
            logger.info(f"start loading embeddings from {embeddings_path}")
            self.table_embeddings = torch.load(embeddings_path)
            logger.info(f"loaded embeddings")

    def retrieve(self, all_queries: list[QueryData]) -> list[list[SearchHit]]:
        all_results = []
        for query in tqdm.tqdm(all_queries, "SBERTRetriever"):
            query_embedding = self.embedder.encode(query.query_text, convert_to_tensor=True, show_progress_bar=False)
            cos_scores = util.cos_sim(query_embedding, self.table_embeddings)[0]
            retrieved = torch.topk(cos_scores, k=self.num_retrieve)

            # search for table id of every retrieved element and find the corresponding score
            # add a tuple containing id and score to the list
            i, top_k_retrieved = 0, []
            for index in retrieved[1]:
                if 0 <= index < len(self.tables):
                    table = self.tables[index]
                    score = retrieved[0][i].item()
                    top_k_retrieved.append(SearchHit(query_id=query.query_id, table_id=table.table_id, score=score))
                    i += 1
            all_results.append(top_k_retrieved)
        return all_results
