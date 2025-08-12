from enum import Enum

import attrs


class Dataset(Enum):
    AITQA = "AITQA"
    NQTABLES = "NQTables"


class QAMode(Enum):
    CLOSED = "closed-book"
    ORACLE = "oracle"
    OPEN = "open-book"


class Linearization(Enum):
    JSON = "json"
    CSV = "csv"


class Template(Enum):
    QTTT = "QTTT"
    TTTQ = "TTTQ"
    QTQTQT = "QTQTQT"


@attrs.define
class TableData:
    table_id: str
    table_name: str
    table_content: str


@attrs.define
class SearchHit:
    query_id: str
    table_id: str
    score: float


@attrs.define
class QueryData:
    query_id: str
    query_text: str
    table_id: str
    ground_truth: list[str] = attrs.field(converter=lambda v: v if isinstance(v, list) else [v])
    generated_answer: str | None


@attrs.define
class RequestData:
    query_id: str
    query_text: str
    qa_mode: QAMode
    context: list[str] | None


@attrs.define
class HitResponsePair:
    search_hit: SearchHit
    response: dict


@attrs.define
class FilteredResponses:
    hit_response_pairs: list[HitResponsePair]
    hits_with_failed_request: list[SearchHit]


@attrs.define
class RetrievalEvalData:
    run: dict[str, dict[str, float]]
    qrels: dict[str, dict[str, int]]


@attrs.define
class QAEvalData:
    predictions: list[dict[str, str]]
    references: list[dict[str, str | dict[str, list]]]


@attrs.define
class QAResults:
    exact_match: float
    relaxed_match: float
    f1_score: float


@attrs.define
class EvalData:
    cutoff_context_ctr: int
    processed_queries_ctr: int
    processed_queries: list[QueryData] | None
    retrieval_eval_data: RetrievalEvalData
    qa_eval_data: QAEvalData
    retrieval_results: dict[str, float] | None
    qa_results: QAResults | None
    total_llm_cost: float
    config: dict
