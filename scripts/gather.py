import json
import logging
import pathlib

import attrs
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from lib.data import get_data_path
from lib.models import Dataset, EvalData

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    dataset: Dataset = MISSING
    partition: str = MISSING
    exp: str | None = None


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    runs_dir = get_data_path() / cfg.dataset.value / "experiments" / cfg.partition / "runs"
    run_paths = list(sorted(runs_dir.glob("*.json")))

    results = pd.DataFrame({"path": run_paths})
    results["eval_data"] = results["path"].apply(load_eval_data)
    results["config_name"] = results["eval_data"].apply(lambda ed: ed["config"]["config_name"])
    component_names = get_component_names(results["config_name"].to_list())
    for component in component_names:
        results[component] = results["config_name"].apply(lambda cn: get_component_value(cn, component))
    results["next_focus_sizes"] = results["eval_data"].apply(get_next_focus_sizes)
    for ret_metric in ("Accuracy@1", "Accuracy@3", "Accuracy@10", "Accuracy@100"):  # , "Accuracy@5"
        results[ret_metric] = results["eval_data"].apply(lambda ed: round(ed["retrieval_results"][ret_metric], 2))
    for qa_metric in ("exact_match", "f1_score"):
        results[qa_metric] = results["eval_data"].apply(lambda ed: round(ed["qa_results"][qa_metric], 2))
    results["llm_cost (ct)"] = results["eval_data"].apply(lambda ed: round(ed["total_llm_cost"] * 100, 2))
    del results["path"]
    del results["config_name"]
    del results["eval_data"]
    if cfg.exp is not None:
        results = results[results["exp"] == cfg.exp]
    results.sort_values(results.columns.to_list(), ascending=True, inplace=True)

    match cfg.exp:
        case "tqa":
            results = results.pivot(
                index=["retriever", "reranker", "zoom", "num_context"],
                columns=["model"],
                values=["exact_match", "llm_cost (ct)"]
            )
            results = results.reindex(("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"), axis=1, level=1)
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv")
        case "retrieval":
            del results["exp"]
            del results["generator"]
            del results["model"]
            del results["num_context"]
            del results["next_focus_sizes"]
            del results["exact_match"]
            del results["f1_score"]
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv", index=False)
        case "zoom":
            del results["exp"]
            del results["retriever"]
            del results["reranker"]
            del results["zoom"]
            del results["generator"]
            del results["linearization"]
            del results["model"]
            del results["num_context"]
            del results["exact_match"]
            del results["f1_score"]
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv", index=False)
        case "template":
            results = results.pivot(
                index=["template"],
                columns=["model"],
                values=["exact_match"]
            )
            results = results.reindex(("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"), axis=1, level=1)
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv")
        case "linear":
            results = results.pivot(
                index=["linearization"],
                columns=["model"],
                values=["exact_match"]
            )
            results = results.reindex(("gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"), axis=1, level=1)
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv")
        case "metadata":
            del results["exp"]
            del results["generator"]
            del results["model"]
            del results["num_context"]
            del results["template"]
            del results["next_focus_sizes"]
            del results["exact_match"]
            del results["f1_score"]
            results.to_csv(get_data_path() / f"{cfg.exp}_{cfg.dataset.value}.csv", index=False)


def load_eval_data(path: pathlib.Path) -> EvalData:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_component_names(config_names: list[str]) -> list[str]:
    component_names = []
    for config_name in config_names:
        for name in [part[:part.index("=")] for part in config_name.split()]:
            if name not in component_names:
                component_names.append(name)
    return component_names


def get_component_value(config_name: str, component_name: str) -> str | None:
    parts = config_name.split()
    for part in parts:
        if part.startswith(f"{component_name}="):
            if "." in part and "lib" in part:
                left = part.rindex(".") + 1
            else:
                left = part.rindex("=") + 1
            return part[left:]
    return None


def get_next_focus_sizes(ed: dict) -> str:
    if "next_focus_sizes" not in ed["config"]["zoom"].keys():
        return ""
    parts = []
    for next_focus_size in ed["config"]["zoom"]["next_focus_sizes"]:
        parts.append(f"-{next_focus_size['num_rows']}rows-> {next_focus_size['num_tables']}")
    return " ".join(parts)


if __name__ == "__main__":
    main()
