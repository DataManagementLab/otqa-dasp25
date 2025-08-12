# Towards Complex Table Question Answering Over Tabular Data Lakes [Extended Version]

## Setup

Make sure you have **[Python 3.11](https://www.python.org)** installed.

Create a virtual environment, activate it, install the dependencies, and add the project to the Python path:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:./
```

## Reproducibility

Reproducing the exact results from the paper requires the following artifacts:

* `llm_cache.zip` the LLM API requests and responses, which you must unpack into `data/llm_cache`
* `AITQA.zip` the AIT-QA dataset, with de-contextualized tables, which you must unpack into `data/AITQA`
* `NQTables.zip` the NQ-Tables dataset, split into easy and hard questions, which you must unpack into `data/NQTables`

To reproduce the results from the paper, run:

```bash
bash reproduce.sh
```

The results are:

* Table 1 (Table question answering results): `data/tqa_AITQA.csv` and `data/tqa_NQTables.csv`
* Table 2 (Table retrieval results): `data/retrieval_AITQA.csv` and `data/retrieval_NQTables.csv`
* Table 3 (Zoom retrieval ablations): `data/zoom_AITQA.csv` and `data/zoom_NQTables.csv`
* Table 4 (Prompt template ablations): `data/template_AITQA.csv` and `data/template_NQTables.csv`
* Table 5 (CSV linearization ablations): `data/linear_AITQA.csv` and `data/linear_NQTables.csv`
* Table 6 (Metadata during retrieval, not in the paper): `data/metadata_AITQA.csv` and `data/metadata_NQTables.csv`
