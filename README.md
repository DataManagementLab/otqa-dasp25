# Towards Complex Table Question Answering Over Tabular Data Lakes (Extended Version)

**Natural Language Interfaces for Databases (NLIDBs) offer an interesting alternative to SQL since they empower 
non-experts to query data. However, they require this data to be integrated into a database schema, causing high 
data engineering and integration overheads. As such, Open Table Question Answering (OTQA) is promising since it 
allows directly querying tables in data lakes without first incorporating them into a relational schema. Many recent 
OTQA approaches combine retrieval-augmented generation with Large Language Models (LLMs), where relevant tables are 
first retrieved from a data lake and then used as input to an LLM to answer the user query. In this paper, we 
systematically analyze how LLMs paired with table retrievers can answer queries over private tabular data lakes. We 
find that the answer generation often fails because the retrieval step does not provide the required tabular context.
To overcome this issue, we propose a novel LLM-based retrieval approach called Zoom retrieval, which effectively 
boosts retrieval accuracies and thereby improves question answering results. Nevertheless, LLMs often still fail to 
answer even simple extraction queries, let alone aggregates, thus remaining far from the rich querying capabilities 
that NLIDBs offer today. Therefore, future work should focus on improving the query execution capabilities of LLMs 
to enable complex question answering over tabular data lakes.**

Please check out our [paper](https://doi.org/10.1007/s13222-025-00513-9) and cite our work:

```bibtex
@article{DBLP:journals/dbsk/RisisBUB25,
    author       = {Daniela Risis and
                  Jan{-}Micha Bodensohn and
                  Matthias Urban and
                  Carsten Binnig},
    title        = {Towards Complex Table Question Answering Over Tabular Data Lakes (Extended
                  Version)},
    journal      = {Datenbank-Spektrum},
    volume       = {25},
    number       = {3},
    pages        = {145--152},
    year         = {2025},
    url          = {https://doi.org/10.1007/s13222-025-00513-9},
    doi          = {10.1007/S13222-025-00513-9}
}
```

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
