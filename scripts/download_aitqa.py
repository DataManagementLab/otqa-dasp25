import logging
import subprocess

import attrs
import hydra
from hydra.core.config_store import ConfigStore

from lib.data import get_data_path
from lib.models import Dataset

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    url: str = "git@github.com:IBM/AITQA.git"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    data_path = get_data_path()
    repo_dir = data_path / Dataset.AITQA.value

    if not repo_dir.is_dir():
        try:
            subprocess.check_call(["git", "clone", cfg.url, str(repo_dir)])
        except Exception as e:
            logger.error("Error while cloning the git repository.")
            logger.error(f"Please manually execute `git clone {cfg.url} {repo_dir}` and restart the script!")
            raise e


if __name__ == "__main__":
    main()
