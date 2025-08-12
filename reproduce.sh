#!/bin/bash

set -e

if [ ! -d "data/llm_cache" ]; then
    echo "You must manually obtain \`data/llm_cache\` (see README.md)!"
    exit
fi

if [ ! -d "data/AITQA" ]; then
    echo "You must manually obtain \`data/AITQA\` (see README.md)!"
    exit
fi

if [ ! -d "data/NQTables" ]; then
    echo "You must manually obtain \`data/NQTables\` (see README.md)!"
    exit
fi

########################################################################################################################
# DOWNLOAD DATASETS
########################################################################################################################

# To reproduce the exact results from our paper, you must use the artifacts `AITQA.zip` and `NQTables.zip` instead of
# downloading the datasets since we have tweaked the datasets as described in the paper. We manually applied our
# de-contextualization approach to the tables of AIT-QA. We split the NQ-Tables dataset into easy and hard questions
# at the beginning of our experiments based on whether GPT-4o-Mini could answer them in a closed-book setting, but have
# since improved our answer generation approach.

# python scripts/download_aitqa.py
# python scripts/download_nqtables.py

########################################################################################################################
# TQA EXPERIMENTS
########################################################################################################################

exp="tqa"
num_context="3"
num_context_high="100"
generator="llm_generator"
linearization="CSV"
template="QTTT"
limit_queries=null

datasets=(
"AITQA"
"NQTABLES"
)

models=(
"gpt-4o-mini-2024-07-18"
"gpt-4o-2024-11-20"
)

retrievers=(
"no_retriever"
"oracle_retriever"
"bm25_retriever"
"sbert_retriever"
)

zooms=(
"no_zoom"
"rowbased_zoom"
)

rerankers=(
"no_reranker"
"query_based_reranker"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi

  for model in "${models[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for zoom in "${zooms[@]}"; do
        for reranker in "${rerankers[@]}"; do
          if [[ "$dataset" == "NQTABLES" && "$model" == "gpt-4o-2024-11-20" ]]; then
            limit_queries=300
          fi
          if [[ "$zoom" == "no_zoom" || "$reranker" == "no_reranker" ]]; then
            if [[ "$retriever" == "bm25_retriever" || "$retriever" == "sbert_retriever" || ( "$reranker" == "no_reranker" &&  "$zoom" == "no_zoom" ) ]]; then
              python scripts/experiment.py \
                'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} template=${template}"' \
                exp=$exp \
                dataset=$dataset \
                partition=$partition \
                retriever=$retriever \
                zoom=$zoom \
                reranker=$reranker \
                generator=$generator \
                llm_model_name=$model \
                num_context=$num_context \
                linearization=$linearization \
                template=$template \
                limit_queries=$limit_queries
            fi
            if [[ ( "$retriever" == "bm25_retriever" ||  "$retriever" == "sbert_retriever" ) && ( "$reranker" == "no_reranker" && "$zoom" == "no_zoom" ) ]]; then
              python scripts/experiment.py \
                'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} template=${template}"' \
                exp=$exp \
                dataset=$dataset \
                partition=$partition \
                retriever=$retriever \
                zoom=$zoom \
                reranker=$reranker \
                generator=$generator \
                llm_model_name=$model \
                num_context=$num_context_high \
                linearization=$linearization \
                template=$template \
                limit_queries=$limit_queries
            fi
          fi
        done
      done
    done
  done
  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done

########################################################################################################################
# RETRIEVAL EXPERIMENTS
########################################################################################################################

exp="retrieval"
model="gpt-4o-mini-2024-07-18"
num_context="0"
generator="no_generator"
template="QTTT"

datasets=(
"AITQA"
"NQTABLES"
)

linearizations=(
"JSON"
"CSV"
)

retrievers=(
"bm25_retriever"
"sbert_retriever"
)

zooms=(
"no_zoom"
"rowbased_zoom"
)

rerankers=(
"no_reranker"
"query_based_reranker"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi

  for linearization in "${linearizations[@]}"; do
    for retriever in "${retrievers[@]}"; do
      for zoom in "${zooms[@]}"; do
        for reranker in "${rerankers[@]}"; do
          if [[ ( "$zoom" == "no_zoom" || "$reranker" == "no_reranker" ) && ( "$linearization" != "JSON" || ( "$zoom" == "no_zoom"  &&  "$reranker" == "no_reranker" ) ) ]]; then
            python scripts/experiment.py \
              'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
              exp=$exp \
              dataset=$dataset \
              partition=$partition \
              retriever=$retriever \
              zoom=$zoom \
              reranker=$reranker \
              generator=$generator \
              llm_model_name=$model \
              num_context=$num_context \
              linearization=$linearization \
              template=$template
          fi
        done
      done
    done
  done
  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done

########################################################################################################################
# ZOOM ABLATION EXPERIMENTS
########################################################################################################################

exp="zoom"
model="gpt-4o-mini-2024-07-18"
num_context="0"
generator="no_generator"
linearization="CSV"
retriever="sbert_retriever"
reranker="no_reranker"
template="QTTT"

datasets=(
"AITQA"
"NQTABLES"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi
  python scripts/experiment.py \
    'config_name="exp=${exp} next_focus_sizes=- retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom="no_zoom" \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    template=$template \
    linearization=$linearization

  python scripts/experiment.py \
    'config_name="exp=${exp} next_focus_sizes=3.1 retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom="rowbased_zoom" \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template \
    'zoom.next_focus_sizes=[{num_tables:3,num_rows:1}]'

  python scripts/experiment.py \
    'config_name="exp=${exp} next_focus_sizes=3.2 retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom="rowbased_zoom" \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template \
    'zoom.next_focus_sizes=[{num_tables:3,num_rows:2}]'

  python scripts/experiment.py \
    'config_name="exp=${exp} next_focus_sizes=10.1-3.5 retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom="rowbased_zoom" \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template \
    'zoom.next_focus_sizes=[{num_tables:10,num_rows:1},{num_tables:3,num_rows:5}]'

  python scripts/experiment.py \
    'config_name="exp=${exp} next_focus_sizes=10.2-3.10 retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization}"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom="rowbased_zoom" \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template \
    'zoom.next_focus_sizes=[{num_tables:10,num_rows:2},{num_tables:3,num_rows:10}]'

  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done

########################################################################################################################
# PROMPT TEMPLATE EXPERIMENTS
########################################################################################################################

exp="template"
num_context="3"
generator="llm_generator"
retriever="sbert_retriever"
zoom="no_zoom"
reranker="no_reranker"
linearization="CSV"
limit_queries=null

datasets=(
"AITQA"
"NQTABLES"
)

models=(
"gpt-4o-mini-2024-07-18"
"gpt-4o-2024-11-20"
)

templates=(
"QTTT"
"TTTQ"
"QTQTQT"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi

  for model in "${models[@]}"; do
    for template in "${templates[@]}"; do
      if ([[ "$dataset" == "NQTABLES" ]] && [[ "$model" == "gpt-4o-2024-11-20" ]]); then
        limit_queries=300
      fi
      python scripts/experiment.py \
        'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} template=${template}"' \
        exp=$exp \
        dataset=$dataset \
        partition=$partition \
        retriever=$retriever \
        zoom=$zoom \
        reranker=$reranker \
        generator=$generator \
        llm_model_name=$model \
        num_context=$num_context \
        linearization=$linearization \
        template=$template \
        limit_queries=$limit_queries
    done
  done
  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done

########################################################################################################################
# LINEARIZATION EXPERIMENTS
########################################################################################################################

exp="linear"
num_context="3"
generator="llm_generator"
retriever="oracle_retriever"
zoom="no_zoom"
reranker="no_reranker"
template="QTTT"
limit_queries=null

datasets=(
"AITQA"
"NQTABLES"
)

models=(
"gpt-4o-mini-2024-07-18"
"gpt-4o-2024-11-20"
)

linearizations=(
"JSON"
"CSV"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi

  for model in "${models[@]}"; do
    for linearization in "${linearizations[@]}"; do
      if [[ "$dataset" == "NQTABLES" && "$model" == "gpt-4o-2024-11-20" ]]; then
        limit_queries=300
      fi
        python scripts/experiment.py \
          'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} template=${template}"' \
          exp=$exp \
          dataset=$dataset \
          partition=$partition \
          retriever=$retriever \
          zoom=$zoom \
          reranker=$reranker \
          generator=$generator \
          llm_model_name=$model \
          num_context=$num_context \
          linearization=$linearization \
          template=$template \
          limit_queries=$limit_queries
    done
  done
  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done


########################################################################################################################
# BM25 WITH(OUT) CONSIDERING METADATA SEPARATELY
########################################################################################################################

exp="metadata"
model="gpt-4o-mini-2024-07-18"
num_context="0"
retriever="bm25_retriever"
zoom="no_zoom"
reranker="no_reranker"
generator="no_generator"
template="QTTT"
linearization="CSV"

datasets=(
"AITQA"
"NQTABLES"
)

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "AITQA" ]]; then
      partition="merged_train_test"
  elif [[ "$dataset" == "NQTABLES" ]]; then
      partition="test/hard"
  else
      echo "Unknown dataset: $dataset"
      exit
  fi

  python scripts/experiment.py \
    'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} metadata=separate"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom=$zoom \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template
  python scripts/experiment.py \
    'config_name="exp=${exp} retriever=${retriever._target_} reranker=${reranker._target_} zoom=${zoom._target_} generator=${generator._target_} model=${llm_model_name} num_context=${num_context} linearization=${linearization} metadata=not-separate"' \
    exp=$exp \
    dataset=$dataset \
    partition=$partition \
    retriever=$retriever \
    zoom=$zoom \
    reranker=$reranker \
    generator=$generator \
    llm_model_name=$model \
    num_context=$num_context \
    linearization=$linearization \
    template=$template \
    retriever.metadata_separate=false

  python scripts/gather.py dataset=$dataset partition=$partition exp=$exp
done
