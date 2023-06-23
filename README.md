# Concept-aware Training

This repository contains training and evaluation sources to train in-context few-shot learners
to utilize concepts in prediction.

Before reproducing the training, note that we make the CoAT-trained models publicly available.
If you simply want to reproduce our results, proceed to **Evaluation** section below and pick the model of your interest.
---
The training of concept-aware model can be reproduced by running the following scripts.

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r training/requirements.txt
pip install -r evaluation/requirements.txt

cd training
./download_teaberac_data.sh
cd ..

CUDA_VISIBLE_DEVICES=0 python training/train_mt5_teabreac+qa_coat.py
```

The script intentionally contains all parameters fixed, but if you need to change something,
e.g. due to the environment restrictions, do not hesitate to adjust `AdaptationArguments` or evaluations within the code.

The training scripts include evaluations on SuperGLUE and various TeaBReaC concepts.


### Baseline: Random Demonstrations Selection Training

In the sequence above, replace the python script path with `train_mt5_teabreac+qa_random.py`.

```shell
CUDA_VISIBLE_DEVICES=0 python training/train_mt5_teabreac+qa_random.py
```

## Evaluations

We provide following pre-trained models from the paper: 
* `Tk-CoAT-1B` corresponds to `authoranonymous321/mt5_large-teabreac-AQA_hard`
* `Tk-CoAT-3B` corresponds to `authoranonymous321/mt5_3B-teabreac-AQA_CoAT`
* `Tk-Random-1B` corresponds to `authoranonymous321/mt5_large-teabreac-AQA_random`
* `Tk-CoAT-1B` corresponds to `authoranonymous321/mt5_3B-teabreac-AQA_random`
* `Tk-Info-3B` corresponds to `authoranonymous321/mt5_3B-teabreac-AQA_informative`

## SuperGLUE evaluation

To reproduce our evaluation on SuperGLUE, run the following:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
CUDA_VISIBLE_DEVICES=0 python evaluation/superglue_evaluator.py \
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard,allenai/tk-instruct-large-def-pos \
    --metric Accuracy \
    --tasks axb,boolq,cb,wsc,copa,multirc,rte,wic,record,axg
```
All resources should be resolved automatically.

## Concept-learning evaluation

To extract the concepts from explanations as proposed in the paper, 
and run the Concept-learning evaluation on a selected model, 
run `sensitivity_evaluator.py` script:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r evaluation/requirements.txt
spacy download en_core_web_sm  # For OpenBookQA concepts extraction

CUDA_VISIBLE_DEVICES=0 python evaluation/sensitivity_evaluator.py \
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard \
    --bootstrap True \
    --metric ROUGE \
    --tasks glue/mnli,openbookqa/additional,hotpot_qa/fullwiki,worldtree \
    --firstn 100
```
All resources and concepts extractions should be resolved automatically.

If you evaluate using `--bootstrapping True`, collect the stdout to a file and analyse the results using [this notebook](analyses/coat_per_prompt_informative_shifts.ipynb).

In order to perform evaluation on a full dataset, simply remove `--firstn` parameter.

## Semantic priors evaluation

To evaluate models' reliance on their semantic representation of labels, 
run the `semantic_priors_evaluator.py` script:

```shell
cd {this_repo}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r evaluation/requirements.txt

CUDA_VISIBLE_DEVICES=0 python evaluation/semantic_priors_evaluator.py \
    --model_names_or_paths authoranonymous321/mt5_large-teabreac-AQA_hard \
    --bootstrap True \
    --aggregate_results True \
    --metric ROUGE \
    --tasks axb,boolq,cb,wsc,multirc,rte,wic,axg \
    --firstn 1000
```

With `--bootstrap True` and `--aggregate_results False`, the results can be vizualized using [this notebook](analyses/coat_priors_reliance_evaluation.ipynb).
To assess the results directly, use `--aggregate_results True` instead.