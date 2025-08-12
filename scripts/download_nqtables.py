import logging
import shutil
import subprocess

import attrs
import hydra
from hydra.core.config_store import ConfigStore

from lib.data import get_data_path
from lib.models import Dataset

logger = logging.getLogger(__name__)


@attrs.define
class Config:
    gs_path: str = "gs://tapas_models/2021_07_22/nq_tables"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    data_path = get_data_path()

    try:
        subprocess.check_call(["gsutil", "-m", "cp", "-R", cfg.gs_path, str(data_path)])
    except Exception as e:
        logger.error("Error while copying data from Google Cloud Storage.")
        logger.error(f"Please manually execute `gsutil -m cp -R {cfg.gs_path} {data_path}` and restart the script!")
        raise e

    shutil.move(data_path / "nq_tables", data_path / Dataset.NQTABLES.value)


if __name__ == "__main__":
    main()
