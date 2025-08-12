########################################################################################################################
# LLM API helpers version: 2025-05-09
########################################################################################################################
import abc
import collections
import contextlib
import dataclasses
import hashlib
import itertools
import json
import logging
import multiprocessing
import multiprocessing.managers
import multiprocessing.pool
import os
import pathlib
import threading
import time
import typing

import requests
import tqdm

from lib.data import get_data_path

logger = logging.getLogger(__name__)

CACHE_PATH: pathlib.Path = get_data_path() / "llm_cache"

MODELS: dict[str, "ModelInfo"] = {
    ####################################################################################################################
    # OpenAI: see https://platform.openai.com/docs/models and https://openai.com/api/pricing/
    ####################################################################################################################
    "gpt-4o-2024-11-20": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 2.50, "batch": 1.25, "flex": 1.25},
        "usd_per_1m_input_tokens_cached": {"default": 1.25, "batch": 1.25, "flex": 1.25},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 10.00, "batch": 5.00, "flex": 5.00}
    },
    "gpt-4o-mini-2024-07-18": {  # added 2025-04-30
        "provider": "openai",
        "max_context": 128_000,
        "max_output_tokens": 16_384,
        "usd_per_1m_input_tokens": {"default": 0.15, "batch": 0.075, "flex": 0.075},
        "usd_per_1m_input_tokens_cached": {"default": 0.075, "batch": 0.075, "flex": 0.075},  # batch and flex unclear
        "usd_per_1m_output_tokens": {"default": 0.60, "batch": 0.30, "flex": 0.30}
    }
}

BUDGET: float | None = 10.0  # budget per model provider, `None` means always execute!
RESPONSES: list = []

NUM_EXEC_THREADS: int = 300
NUM_TOKEN_PROCESSES: int = 16

PROVIDERS: dict[str, "BaseProvider"] = {}


class Request(typing.TypedDict):
    model: str


class Response(typing.TypedDict):
    model: str


def execute(
        request_or_requests: Request | list[Request],
        /, *,
        budget: float | None = 0.0,
        batch: bool = False,
        silent: bool = False,
        auto_transform: str | None = None
) -> Response | list[Response]:
    """Execute one request synchronously or multiple requests in parallel.

    Args:
        request_or_requests: The request or list of requests to execute.
        budget: Execute without confirmation if costs are below budget. `None` means always execute!
        batch: Execute requests as batches (this can take a long time and needs care).
        silent: Disable log messages and process bars.
        auto_transform: Automatically transform requests and responses.

    Returns:
        The response or list of responses.
    """
    global RESPONSES
    if request_or_requests == []:
        return []

    provider = BaseProvider.check_determine_provider(request_or_requests)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    request_or_requests = provider.transform_from(request_or_requests, auto_provider)

    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    if batch:
        assert isinstance(request_or_requests, list), "cannot batch individual requests"
        if not silent:
            with ProgressBar(total=len(request_or_requests), disable=silent) as progress_bar:
                response_or_responses = provider.execute_batch(
                    request_or_requests,
                    budget,
                    silent,
                    progress_bar
                )
        else:
            response_or_responses = provider.execute_batch(request_or_requests, budget, silent, None)
    elif isinstance(request_or_requests, dict):  # dict = Request
        # noinspection PyTypeChecker
        response_or_responses = provider.execute([request_or_requests], budget, silent, None)[0]
    elif isinstance(request_or_requests, list):
        if not silent:
            with ProgressBar(total=len(request_or_requests)) as progress_bar:
                response_or_responses = provider.execute(request_or_requests, budget, silent, progress_bar)
        else:
            response_or_responses = provider.execute(request_or_requests, budget, silent, None)
    else:
        typing.assert_never(request_or_requests)
    responses = provider.transform_to(response_or_responses, auto_provider)
    RESPONSES += responses
    return responses


def count_tokens(
        request_or_requests: Request | list[Request],
        /, *,
        silent: bool = False,
        auto_transform: str | None = None
) -> int | list[int]:
    """Count the number of input tokens for one request synchronously or for multiple requests in parallel.

    Args:
        request_or_requests: The request or list of requests for which to count input tokens.
        silent: Disable log messages and process bars.
        auto_transform: Automatically transform requests.

    Returns:
        The number of tokens or list of numbers of tokens.
    """
    if request_or_requests == []:
        return []

    provider = BaseProvider.check_determine_provider(request_or_requests)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    request_or_requests = provider.transform_from(request_or_requests, auto_provider)

    if isinstance(request_or_requests, dict):  # dict = Request
        # noinspection PyTypeChecker
        return provider.count_tokens([request_or_requests], silent, None)[0]
    elif isinstance(request_or_requests, list):
        if not silent:
            with ProgressBar(total=len(request_or_requests), disable=silent) as progress_bar:
                return provider.count_tokens(request_or_requests, silent, progress_bar)
        else:
            return provider.count_tokens(request_or_requests, silent, None)
    else:
        typing.assert_never(request_or_requests)


def count_tokens_str(
        string_or_strings: str | list[str],
        model: str,
        /, *,
        silent: bool = False
) -> int | list[int]:
    """Count the number of input tokens for one string synchronously or for multiple strings in parallel.

    Args:
        string_or_strings: The string or list of strings for which to count tokens.
        model: The name of the model.
        silent: Disable log messages and process bars.

    Returns:
        The number of tokens or list of numbers of tokens.
    """
    if string_or_strings == []:
        return []

    provider = BaseProvider.check_determine_provider(model)

    if isinstance(string_or_strings, str):
        return provider.count_tokens_str([string_or_strings], model, silent, None)[0]
    elif isinstance(string_or_strings, list):
        if not silent:
            with ProgressBar(total=len(string_or_strings), disable=silent) as progress_bar:
                return provider.count_tokens_str(string_or_strings, model, silent, progress_bar)
        else:
            return provider.count_tokens_str(string_or_strings, model, silent, None)
    else:
        typing.assert_never(string_or_strings)


def cost(
        response_or_responses: Response | list[Response],
        /, *,
        ignore_token_caching: bool = False,
        batch: bool = False,
        silent: bool = False,
        auto_transform: str | None = None
) -> float | list[float]:
    """Compute the USD costs for one or multiple responses.

    Args:
        response_or_responses: The response or list of responses.
        ignore_token_caching: Ignore that some tokens were cached and therefore cheaper.
        batch: Assume that requests were executed as batches.
        silent: Disable log messages and process bars.
        auto_transform: Automatically transform responses.

    Returns:
        The USD cost or list of costs.
    """
    if response_or_responses == []:
        return []

    provider = BaseProvider.check_determine_provider(response_or_responses)
    auto_provider = BaseProvider.check_determine_auto_provider(auto_transform)
    if auto_provider is not None:
        response_or_responses = auto_provider.transform_to(response_or_responses, provider)

    if isinstance(response_or_responses, dict):  # dict = Response
        # noinspection PyTypeChecker
        return provider.cost([response_or_responses], ignore_token_caching, batch, silent, None)[0]
    elif isinstance(response_or_responses, list):
        if not silent:
            with ProgressBar(total=len(response_or_responses), disable=silent) as progress_bar:
                return provider.cost(
                    response_or_responses,
                    ignore_token_caching,
                    batch,
                    silent,
                    progress_bar
                )
        else:
            return provider.cost(response_or_responses, ignore_token_caching, batch, silent, None)
    else:
        typing.assert_never(response_or_responses)


@contextlib.contextmanager
def client():
    """Client context manager to orchestrate API calls when using multiprocessing."""
    with multiprocessing.Manager() as manager:
        for provider in PROVIDERS.values():
            provider.state.migrate(manager.dict(), manager.Semaphore(), manager.Semaphore())
        try:
            yield
        finally:
            for provider in PROVIDERS.values():
                provider.state.migrate({}, threading.Semaphore(), threading.Semaphore())


########################################################################################################################
# generic implementation
########################################################################################################################

class CostInfo(typing.TypedDict, total=False):
    default: float
    batch: float
    flex: float


class ModelInfo(typing.TypedDict, total=False):
    provider: str
    max_context: int
    max_output_tokens: int
    usd_per_1m_input_tokens: CostInfo
    usd_per_1m_input_tokens_cached: CostInfo
    usd_per_1m_output_tokens: CostInfo
    hf_model_name: str


@dataclasses.dataclass
class State:
    state: dict | multiprocessing.managers.DictProxy
    state_semaphore: threading.Semaphore
    wait_semaphore: threading.Semaphore

    def migrate(
            self,
            state: dict | multiprocessing.managers.DictProxy,
            state_semaphore: threading.Semaphore,
            wait_semaphore: threading.Semaphore
    ) -> None:
        with state_semaphore:
            with self.state_semaphore:
                for k, v in self.state.items():
                    state[k] = v
                self.state = state
                self.state_semaphore = state_semaphore
                self.wait_semaphore = wait_semaphore


class ProgressBar(tqdm.tqdm):
    cached: int
    running: int
    failed: int
    cost: float
    bottleneck: typing.Literal["P"] | typing.Literal["S"] | typing.Literal["L"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cached = 0
        self.failed = 0
        self.cost = 0
        self.bottleneck = "P"
        self.update_postfix()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return super().__exit__(exc_type, exc_value, traceback)

    def next_step(self, desc: str, left: int) -> None:
        super().reset(total=self.total)
        self.set_description(desc)
        self.update(self.total - left)

    def update(self, *args, **kwargs) -> None:
        self.update_postfix()
        super().update(*args, **kwargs)

    def update_postfix(self) -> None:
        bottleneck = "L" if self.bottleneck == "L" else None
        failed = f"failed={self.failed}" if self.failed > 0 else None
        cached = f"cached={self.cached}" if self.cached > 0 else None
        cost_ = f"cost=${self.cost:.2f}" if self.cost > 0 else None

        self.set_postfix_str(", ".join(p for p in (bottleneck, failed, cached, cost_) if p is not None))


@dataclasses.dataclass
class BasePair(abc.ABC):
    request: Request
    response: Response | None = None


class BaseProvider(abc.ABC):
    name: str = "base-provider"
    trans_warning_cache: set
    state: State

    def __init__(self) -> None:
        self.trans_warning_cache = set()
        self.state = State({}, threading.Semaphore(), threading.Semaphore())

    @classmethod
    def check_determine_model(cls, value: str | Request | list[Request] | Response | list[Response]) -> str:
        if isinstance(value, str):
            model = value
        elif isinstance(value, dict):  # dict = Request | Response
            model = value["model"]
        elif isinstance(value, list):
            models = set(r["model"] for r in value)
            assert len(models) == 1, "all requests/responses must use the same model"
            (model,) = models
        else:
            typing.assert_never(value)
        assert model in MODELS.keys(), f"unknown model `{model}`"
        return model

    @classmethod
    def check_determine_provider(cls, value: str | Request | list[Request] | Response | list[Response]) -> typing.Self:
        model = cls.check_determine_model(value)
        provider = MODELS[model]["provider"]
        assert provider in PROVIDERS.keys(), f"unknown model provider `{provider}`"
        return PROVIDERS[provider]

    @classmethod
    def check_determine_auto_provider(cls, auto_transform: str | None) -> typing.Self | None:
        if auto_transform is not None:
            assert auto_transform in PROVIDERS.keys(), f"unknown model provider `{auto_transform}` for auto_transform"
            return PROVIDERS[auto_transform]
        return None

    def transform_from(
            self,
            request_or_requests: Request | list[Request],
            from_provider: typing.Self | None
    ) -> Request | list[Request]:
        if from_provider is None:
            return request_or_requests
        elif isinstance(request_or_requests, dict):  # dict = Request
            # noinspection PyTypeChecker
            return self.transform_request_from(request_or_requests, from_provider)
        elif isinstance(request_or_requests, list):
            return [self.transform_request_from(req, from_provider) for req in request_or_requests]
        else:
            typing.assert_never(request_or_requests)

    def transform_to(
            self,
            response_or_responses: Response | list[Response],
            to_provider: typing.Self | None
    ) -> Response | list[Response]:
        if to_provider is None:
            return response_or_responses
        elif isinstance(response_or_responses, dict):  # dict = Response
            # noinspection PyTypeChecker
            return self.transform_response_to(response_or_responses, to_provider)
        elif isinstance(response_or_responses, list):
            return [self.transform_response_to(res, to_provider) for res in response_or_responses]
        else:
            typing.assert_never(response_or_responses)

    def user_confirms_cost(self, cost_: float, budget: float, silent: bool, progress_bar: ProgressBar | None) -> None:
        if progress_bar is not None:
            progress_bar.clear()
        with self.state.state_semaphore:
            if budget is not None and cost_ > budget \
                    or BUDGET is not None and self.state.state["total_cost"] + cost_ > BUDGET:
                logger.info(
                    f"already spent ${self.state.state['total_cost']:.2f}, press enter to continue and spend up to around ${cost_:.2f}")
                input(
                    f"already spent ${self.state.state['total_cost']:.2f}, press enter to continue and spend up to around ${cost_:.2f}")
            elif not silent:
                logger.info(f"already spent ${self.state.state['total_cost']:.2f}, spending up to around ${cost_:.2f}")
            self.state.state["total_cost"] = self.state.state["total_cost"] + cost_

    def compute_hash(self, pair: BasePair) -> str:
        return hashlib.sha256(bytes(f"{self.name}-{json.dumps(pair.request)}", "utf-8")).hexdigest()

    def load_cached_response(self, pair: BasePair) -> bool:
        path = CACHE_PATH / f"{self.compute_hash(pair)}.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as file:
                cached_pair = json.load(file)
            if cached_pair["provider"] == self.name and cached_pair["request"] == pair.request:
                pair.response = cached_pair["response"]
                return True
        return False

    def save_to_cache(self, pair: BasePair) -> None:
        path = CACHE_PATH / f"{self.compute_hash(pair)}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump({
                "provider": self.name,
                "request": pair.request,
                "response": pair.response
            }, file)

    def trans_warning(self, message: str) -> None:
        message_prefix = message[:message.rindex("=")] if "=" in message else message
        if message_prefix not in self.trans_warning_cache:
            logger.warning(message)
            self.trans_warning_cache.add(message_prefix)

    @staticmethod
    def autodict() -> collections.defaultdict:
        return collections.defaultdict(BaseProvider.autodict)

    @classmethod
    def autodict_to_dict(cls, x: typing.Any) -> typing.Any:
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r[k] = cls.autodict_to_dict(v)
            return r
        elif isinstance(x, list):
            return [cls.autodict_to_dict(v) for v in x]
        else:
            return x

    @abc.abstractmethod
    def execute(
            self,
            requests: list[Request],
            budget: float | None,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        raise NotImplementedError()

    @abc.abstractmethod
    def execute_batch(
            self,
            requests: list[Request],
            budget: float | None,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[Response]:
        raise NotImplementedError()

    @abc.abstractmethod
    def count_tokens(self, requests: list[Request], silent: bool, progress_bar: ProgressBar | None) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def cost(
            self,
            responses: list[Response],
            ignore_token_caching: bool,
            batch: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_request_from(self, request: Request, provider: typing.Self) -> Request:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_response_to(self, response: Response, provider: typing.Self) -> Response:
        raise NotImplementedError()


########################################################################################################################
# OpenAI implementation
########################################################################################################################


class OpenAIRequestMessage(typing.TypedDict):
    content: str


class OpenAIRequest(Request, total=False):
    messages: list[OpenAIRequestMessage]
    max_tokens: int | None
    max_completion_tokens: int | None


class OpenAIUsagePromptTokensDetails(typing.TypedDict):
    cached_tokens: int


class OpenAIUsage(typing.TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_tokens_details: OpenAIUsagePromptTokensDetails


class OpenAIResponse(Response):
    usage: OpenAIUsage


@dataclasses.dataclass
class OpenAIRateLimitBudget:
    mode: typing.Literal["sequential"] | typing.Literal["parallel"] = "sequential"
    rpm: float | None = None
    tpm: float | None = None
    r: float | None = None
    t: float | None = None
    last_update: float = dataclasses.field(default_factory=time.time)

    def is_enough_for_request(self, pair: "OpenAIPair") -> bool:
        return (self.r is None or self.r >= 1) and (self.t is None or self.t >= pair.est_max_usage)

    def consider_time(self) -> typing.Self:
        now = time.time()
        delta = now - self.last_update
        if self.rpm is not None and self.r is not None:
            self.r = min(self.rpm, self.r + self.rpm * delta / 60)
        if self.tpm is not None and self.t is not None:
            self.t = min(self.tpm, self.t + self.tpm * delta / 60)
        self.last_update = now
        return self

    def decrease_by_request(self, pair: "OpenAIPair") -> typing.Self:
        if self.r is not None:
            self.r -= 1
        if self.t is not None:
            self.t -= pair.est_max_usage
        return self

    def set_from_headers(self, headers: dict[str, typing.Any]) -> typing.Self:
        if "x-ratelimit-limit-requests" in headers.keys():
            self.rpm = int(headers["x-ratelimit-limit-requests"])
        if "x-ratelimit-limit-tokens" in headers.keys():
            self.tpm = int(headers["x-ratelimit-limit-tokens"])
        if "x-ratelimit-remaining-requests" in headers.keys():
            header_r = int(headers["x-ratelimit-remaining-requests"])
            if self.r is None or self.r > header_r:
                self.r = header_r
        if "x-ratelimit-remaining-tokens" in headers.keys():
            header_t = int(headers["x-ratelimit-remaining-tokens"])
            if self.t is None or self.t > header_t:
                self.t = header_t
        return self

    def to_par(self) -> typing.Self:
        self.mode = "parallel"
        return self

    def to_seq(self) -> typing.Self:
        self.mode = "sequential"
        return self


@dataclasses.dataclass
class OpenAIPair(BasePair):
    request: OpenAIRequest
    response: OpenAIResponse | None = None
    est_max_cost: float | None = None
    est_max_usage: int | None = None

    def set_cost_and_usage(self, est_input_tokens: int) -> None:
        model_params = MODELS[self.request["model"]]

        assert "n" not in self.request.keys() and "best_of" not in self.request.keys(), "`n` and `best_of` not supported"

        if "max_completion_tokens" in self.request.keys() and self.request["max_completion_tokens"] is not None:
            est_max_output_tokens = self.request["max_completion_tokens"]
        elif "max_tokens" in self.request.keys() and self.request["max_tokens"] is not None:
            est_max_output_tokens = self.request["max_tokens"]
        else:
            left_for_output = max(0, model_params["max_context"] - est_input_tokens)
            est_max_output_tokens = min(left_for_output, model_params["max_output_tokens"])
        mode = "default"
        self.est_max_cost = est_input_tokens * model_params["usd_per_1m_input_tokens"][mode] / 1_000_000 \
                            + est_max_output_tokens * model_params["usd_per_1m_output_tokens"][mode] / 1_000_000
        self.est_max_usage = est_input_tokens + est_max_output_tokens

    def execute(self, provider: "OpenAIProvider", progress_bar: ProgressBar | None, has_semaphore: bool) -> bool:
        http_response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            json=self.request,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        )

        with provider.state.state_semaphore if not has_semaphore else contextlib.nullcontext():
            provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].set_from_headers(
                http_response.headers
            )
            match http_response.status_code:
                case 200:
                    self.response = http_response.json()
                    provider.save_to_cache(self)
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_par()
                    cost_ = provider.cost([self.response], False, False, True, None)[0]
                    provider.state.state["total_cost"] = provider.state.state["total_cost"] + cost_ - self.est_max_cost
                    if progress_bar is not None:
                        progress_bar.cost += cost_
                        progress_bar.update()
                    return True
                case 429:
                    logger.info("retry request due to rate limit error")
                    provider.state.state[self.request["model"]] = provider.state.state[self.request["model"]].to_seq()
                    if progress_bar is not None:
                        progress_bar.update_postfix()
                    return False
                case _:
                    logger.warning(f"request failed, no retry: {http_response.content}")
                    try:
                        self.response = http_response.json()
                    except json.decoder.JSONDecodeError:
                        self.response = {"error": http_response.content}
                    if progress_bar is not None:
                        progress_bar.failed += 1
                        progress_bar.update()
                    return True

    def wait_execute_retry(self, provider: "OpenAIProvider", progress_bar: ProgressBar | None) -> None:
        request_done = False
        while not request_done:
            execute_parallel = False
            with provider.state.wait_semaphore:  # make sure this thread will be next to execute
                while not (request_done or execute_parallel):
                    with provider.state.state_semaphore:
                        state = provider.state.state
                        if self.request["model"] not in state.keys():
                            state[self.request["model"]] = OpenAIRateLimitBudget()

                        state[self.request["model"]] = state[self.request["model"]].consider_time()

                        if state[self.request["model"]].is_enough_for_request(self):
                            state[self.request["model"]] = state[self.request["model"]].decrease_by_request(self)

                            match state[self.request["model"]].mode:
                                case "sequential":  # execute inside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "S"
                                        progress_bar.update_postfix()
                                    if self.execute(provider, progress_bar, True):
                                        request_done = True
                                case "parallel":  # execute outside semaphore
                                    if progress_bar is not None:
                                        progress_bar.bottleneck = "P"
                                        progress_bar.update_postfix()
                                    execute_parallel = True
                                case _ as other:
                                    typing.assert_never(other)
                        else:
                            if progress_bar is not None:
                                progress_bar.bottleneck = "L"
                                progress_bar.update_postfix()
                            time.sleep(0.05)  # sleep to wait for rate limit budget

            if execute_parallel:
                if self.execute(provider, progress_bar, False):
                    request_done = True


class OpenAIProvider(BaseProvider):
    name: str = "openai"
    additional_tokens: int = 10  # number of additional tokens per message

    def __init__(self) -> None:
        super().__init__()
        self.state.state["total_cost"] = 0.0

    def execute(
            self,
            requests: list[OpenAIRequest],
            budget: float | None,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[OpenAIResponse]:
        pairs = [OpenAIPair(request) for request in requests]

        if progress_bar is not None:
            progress_bar.next_step("load responses", len(pairs))

        pairs_to_execute = []
        for pair in pairs:
            if not self.load_cached_response(pair):
                pairs_to_execute.append(pair)
            else:
                if progress_bar is not None:
                    progress_bar.cached += 1
            if progress_bar is not None:
                progress_bar.update()

        if len(pairs_to_execute) > 0:
            assert "OPENAI_API_KEY" in os.environ.keys(), "missing `OPENAI_API_KEY` in environment variables"

            # determine maximum cost
            reqs_to_execute = [pair.request for pair in pairs_to_execute]
            est_input_tokenss = self.count_tokens(reqs_to_execute, silent, progress_bar)
            for pair, est_input_tokens in zip(pairs_to_execute, est_input_tokenss):
                pair.set_cost_and_usage(est_input_tokens)
            est_max_total_cost = sum(pair.est_max_cost for pair in pairs_to_execute)
            self.user_confirms_cost(est_max_total_cost, budget, silent, progress_bar)

            # sort request to execute longest first, but put one short request first to quickly obtain HTTP header
            pairs_to_execute.sort(key=lambda p: p.est_max_usage, reverse=True)
            pairs_to_execute = pairs_to_execute[-1:] + pairs_to_execute[:-1]

            # execute requests
            if progress_bar is not None:
                progress_bar.next_step("execute requests", len(pairs_to_execute))

            if len(pairs_to_execute) > 1:
                with multiprocessing.pool.ThreadPool(
                        processes=min(NUM_EXEC_THREADS, len(pairs_to_execute))) as pool:
                    params = [(pair, self, progress_bar) for pair in pairs_to_execute]
                    pool.map(lambda p: p[0].wait_execute_retry(*p[1:]), params)
            else:
                pairs_to_execute[0].wait_execute_retry(self, progress_bar)

        return [pair.response for pair in pairs]

    def execute_batch(
            self,
            requests: list[OpenAIRequest],
            budget: float | None,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[OpenAIResponse]:
        raise NotImplementedError()

    def count_tokens(
            self,
            requests: list[OpenAIRequest],
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(requests))

        model = self.check_determine_model(requests)

        import tiktoken
        encoding = tiktoken.encoding_for_model(model)

        contents, lefts = [], []
        for req in requests:
            lefts.append(len(contents))
            contents += [m["content"] for m in req["messages"]]
        lefts.append(len(contents))

        lengths = [len(tokens) + self.additional_tokens for tokens in
                   encoding.encode_batch(contents, num_threads=NUM_TOKEN_PROCESSES)]
        num_tokens = [sum(lengths[l:r]) for l, r in itertools.pairwise(lefts)]

        if progress_bar is not None:
            progress_bar.update(len(requests))
        return num_tokens

    def count_tokens_str(
            self,
            strings: list[str],
            model: str,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[int]:
        if progress_bar is not None:
            progress_bar.next_step("count tokens", len(strings))

        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = [len(tokens) for tokens in encoding.encode_batch(strings, num_threads=NUM_TOKEN_PROCESSES)]
        if progress_bar is not None:
            progress_bar.update(len(strings))
        return num_tokens

    def cost(
            self,
            responses: list[OpenAIResponse],
            ignore_token_caching: bool,
            batch: bool,
            silent: bool,
            progress_bar: ProgressBar | None
    ) -> list[float]:
        if progress_bar is not None:
            progress_bar.next_step("compute cost", len(responses))

        model = self.check_determine_model(responses)

        mode = "batch" if batch else "default"
        usd_per_1m_input_tokens = MODELS[model]["usd_per_1m_input_tokens"][mode]
        usd_per_1m_input_tokens_cached = MODELS[model]["usd_per_1m_input_tokens_cached"][mode]
        usd_per_1m_output_tokens = MODELS[model]["usd_per_1m_output_tokens"][mode]

        costs = []
        for response in responses:
            cost_ = 0
            if ignore_token_caching:
                cost_ += response["usage"]["prompt_tokens"] * usd_per_1m_input_tokens / 1_000_000
            else:
                cached_tokens = response["usage"]["prompt_tokens_details"]["cached_tokens"]
                cost_ += cached_tokens * usd_per_1m_input_tokens_cached / 1_000_000
                prompt_tokens = response["usage"]["prompt_tokens"] - cached_tokens
                cost_ += prompt_tokens * usd_per_1m_input_tokens / 1_000_000
            cost_ += response["usage"]["completion_tokens"] * usd_per_1m_output_tokens / 1_000_000
            costs.append(cost_)
            if progress_bar is not None:
                progress_bar.update()
        return costs

    def transform_request_from(self, request: Request, provider: BaseProvider) -> OpenAIRequest:
        match provider.name:
            case "openai" | None:
                # noinspection PyTypeChecker
                return request
            case _ as other:
                typing.assert_never(other)

    def transform_response_to(self, response: OpenAIResponse, provider: BaseProvider) -> Response:
        match provider.name:
            case "openai" | None:
                return response
            case _ as other:
                typing.assert_never(other)


PROVIDERS[OpenAIProvider.name] = OpenAIProvider()
