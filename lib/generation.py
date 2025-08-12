import abc
import json
import logging
import typing

import attrs
import tiktoken
from hydra.core.config_store import ConfigStore
from omegaconf import omegaconf

from lib.llms import MODELS, execute, count_tokens_str
from lib.models import Dataset, QAMode, RequestData, Template
from lib.utils import fill_template

logger = logging.getLogger(__name__)
cs = ConfigStore.instance()

prompt_templates: dict[Template, dict[Dataset, dict[QAMode, str]]] = {
    Template.QTTT: {
        Dataset.AITQA: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the query. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Query: {{query}} Context: {{context}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Query: {{query}} Context: {{context}}""",
        },
        Dataset.NQTABLES: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the question. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Query: {{query}} Context: {{context}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Query: {{query}} Context: {{context}}""",
        }
    },
    Template.TTTQ: {
        Dataset.AITQA: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the query. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}} Query: {{query}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}} Query: {{query}}""",
        },
        Dataset.NQTABLES: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the question. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}} Query: {{query}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}} Query: {{query}}""",
        }
    },
    Template.QTQTQT: {
        Dataset.AITQA: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the query. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}}""",
        },
        Dataset.NQTABLES: {
            QAMode.CLOSED: """Answer in following format {"answer": "<your-answer>"}. Use one word or a phrase to answer the question. Query: {{query}}""",
            QAMode.ORACLE: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the table. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}}""",
            QAMode.OPEN: """Answer in following format {"answer": "<your-answer>"}. Answer the query below based on the tables. Only extract the answer from one table cell. Do not generate a sentence. Context: {{context}}""",
        }
    }
}


########################################################################################################################
# BASE GENERATOR
########################################################################################################################

@attrs.define
class BaseGeneratorConfig(abc.ABC):
    _target_: str = omegaconf.MISSING


@attrs.define
class BaseGenerator(abc.ABC):
    dataset: Dataset
    qa_mode: QAMode
    llm_model_name: str
    llm_seed: int
    template: Template
    cutoff_context_ctr: int = 0

    @abc.abstractmethod
    def generate(self, requests_data: list[RequestData]) -> list[str]:
        raise NotImplementedError()


########################################################################################################################
# NO GENERATOR
########################################################################################################################

@attrs.define
class NoGeneratorConfig(BaseGeneratorConfig):
    _target_: str = "lib.generation.NoGenerator"


cs.store(name="no_generator", node=NoGeneratorConfig, group="generator")


@attrs.define
class NoGenerator(BaseGenerator):

    def __attrs_post_init__(self) -> None:
        self.llm_model_name = "none"

    def generate(self, requests_data: list[RequestData]) -> list[str]:
        return ["" for _ in requests_data]


########################################################################################################################
# LLM GENERATOR
########################################################################################################################

@attrs.define
class LLMGeneratorConfig(BaseGeneratorConfig):
    _target_: str = "lib.generation.LLMGenerator"
    buffer_tokens: int = 100
    temperature: float = 0
    max_answer_tokens: int = 70


cs.store(name="llm_generator", node=LLMGeneratorConfig, group="generator")


@attrs.define
class LLMGenerator(BaseGenerator):
    buffer_tokens: int = None  # not so elegant having to set a value here...
    temperature: float = None
    max_answer_tokens: int = None
    max_prompt_tokens: int = 0

    def __attrs_post_init__(self):
        self.max_prompt_tokens = MODELS[self.llm_model_name]["max_context"] - self.max_answer_tokens

    def generate(self, requests_data: list[RequestData]) -> list[str]:
        # construct the list of requests
        requests = [
            {
                "model": self.llm_model_name,
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant that answers questions based on information stored in tables."},
                    {"role": "user",
                     "content": self.format_prompt(request.query_id, request.query_text, request.qa_mode,
                                                   request.context)},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_answer_tokens,
                "seed": self.llm_seed,
            }
            for request in requests_data
        ]

        responses = execute(requests, budget=None, auto_transform="openai")

        return self.parse_responses(responses)

    def format_prompt(self, query_id: str, query_text: str, qa_mode: QAMode, context: list[str] | None) -> str:
        """Format the prompt by first selecting the correct prompt template for the requested question answering mode
        and then adding the given parameters."""
        prompt_template = prompt_templates[self.template][self.dataset][qa_mode]

        match qa_mode:
            case QAMode.CLOSED:
                prompt = fill_template(prompt_template, query=query_text, query_id=query_id)
            case QAMode.ORACLE:
                match self.template:
                    case Template.QTTT | Template.TTTQ:
                        context_str = f"Ground truth table : {context[0]}"
                    case Template.QTQTQT:
                        context_str = f"Query: {query_text} Ground truth table : {context[1]}"
                    case _ as template:
                        typing.assert_never(template)

                context_str = self.validate_context_len(prompt_template, query_text, context_str)
                prompt = fill_template(prompt_template, query_id=query_id, query=query_text, context=context_str)
            case QAMode.OPEN:
                match self.template:
                    case Template.QTTT | Template.TTTQ:
                        context_str = "\n".join(f"Table {i}: {table}" for i, table in enumerate(context, start=1))
                    case Template.QTQTQT:
                        context_str = "\n".join(
                            f"Query: {query_text} Table {i}: {table}" for i, table in enumerate(context, start=1))
                    case _ as template:
                        typing.assert_never(template)
                context_str = self.validate_context_len(prompt_template, query_text, context_str)
                prompt = fill_template(prompt_template, query_id=query_id, query=query_text, context=context_str)
            case _ as qa_mode:
                typing.assert_never(qa_mode)

        return prompt

    def validate_context_len(self, prompt_template: str, query: str, context: str) -> str:
        """Check whether the tabular context of the prompt exceeds the maximum context window of the used LLM."""
        template_len = count_tokens_str(prompt_template, self.llm_model_name)
        query_len = count_tokens_str(query, self.llm_model_name)
        context_len = count_tokens_str(context, self.llm_model_name)

        # the LLM may add some tokens to the prompt for internal purposes
        available_tokens = self.max_prompt_tokens - template_len - query_len - self.buffer_tokens

        if context_len > available_tokens:
            encoding = tiktoken.encoding_for_model(self.llm_model_name)
            tokens = encoding.encode(context)
            truncated_tokens = tokens[:available_tokens]
            context = encoding.decode(truncated_tokens)
            logger.info(f"prompt would exceed maximum token limit, shortened context by "
                        f"{context_len - available_tokens} tokens.")
            self.cutoff_context_ctr += 1

        return context

    def parse_responses(self, responses: list[dict]) -> list[str]:
        parsed_responses = []
        for idx, response in enumerate(responses):
            content = response["choices"][0]["message"]["content"]
            try:
                if content.startswith("```json"):
                    content = content[len("```json"):]
                if content.endswith("```"):
                    content = content[:-len("```")]
                content = content.strip()
                parsed_responses.append(str(json.loads(content)["answer"]))
            except Exception as e:
                logger.warning(f"failed to parse LLM response: {e} in {content}")
                parsed_responses.append("")
        return parsed_responses
