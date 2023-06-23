import math

import numpy as np
import pandas as pd
import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from evaluation.evaluator_informative import RougeInformative, AccuracyInformative

from evaluation.tasks.en.glue_diagnostics import GLUEDiagnostics
import argparse

from evaluation.tasks.en.openbookqa import OpenBookQATask
from evaluation.tasks.en.r4c_hotpotqa import R4CHotpotQATask
from evaluation.tasks.en.teabreac import TeaBReAC
from evaluation.tasks.en.worldtree_qa import WorldTreeQA

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--use_cache", type=str, default="False", choices=('True', 'False'),
                    help="Whether to use cached predictions, if available.")
parser.add_argument("--tasks", default="glue/mnli,openbookqa/additional,hotpot_qa/fullwiki,worldtree,teabreac", type=str,
                    help="Coma-separated list of evaluation datasets. Must be one of the implemented datasets: "
                         "'glue/mnli', 'openbookqa/additional', 'hotpot_qa/fullwiki', 'worldtree', 'teabreac'")
parser.add_argument("--template_names", default=None, type=str,
                    help="Names of the templates to evaluate with")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--bootstrap", default="True", type=str,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--aggregate_results", default="False", type=str,
                    help="Whether to print the results aggregated into confidence intervals, or separate evaluations.")

args = parser.parse_args()

args.bootstrap = args.bootstrap != "False"
args.use_cache = args.use_cache != "False"
args.aggregate_results = args.aggregate_results != "False"

results = {}


for model_name_or_path in args.model_names_or_paths.split(","):
    results[model_name_or_path] = {}
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    for dataset_id in args.tasks.split(","):
        # eval templates resolution
        if args.template_names is not None:
            eval_templates = args.template_names.split(",")
        else:
            if dataset_id == 'hotpot_qa/fullwiki':
                # only two templates for hotpot_qa require answering questions, others are for different tasks
                eval_templates = ['generate_answer_interrogative', 'generate_answer_affirmative']
            else:
                eval_templates = DatasetTemplates(dataset_id).all_template_names
                if not eval_templates:
                    eval_templates = ["no template"]

        for template_id in eval_templates:
            template = DatasetTemplates(dataset_id)[template_id] if template_id != "no template" else None
            # eval task resolution - done in the loop to reset its state (deduplication)
            if dataset_id == "glue/mnli":
                task = GLUEDiagnostics("en", template)
            elif dataset_id == "openbookqa/additional":
                task = OpenBookQATask("en", template)
            elif dataset_id == 'hotpot_qa/fullwiki':
                task = R4CHotpotQATask("en", template)
            elif dataset_id == 'worldtree':
                task = WorldTreeQA("en", template)
            elif dataset_id == 'teabreac':
                task = TeaBReAC(teabreac_val_dataset_jsonl="training/data/teabreac_v1.0_multihop_qa_dev.jsonl")
            else:
                raise ValueError("Non-implemented dataset: %s" % dataset_id)

            # evaluation metric resolution
            if args.metric == "ROUGE":
                evaluator = RougeInformative(task, bootstrap=args.bootstrap, max_input_length=args.max_input_length)
            elif args.metric == "Accuracy":
                evaluator = AccuracyInformative(task, bootstrap=args.bootstrap, max_input_length=args.max_input_length)
            else:
                raise ValueError("Unknown metric: %s" % args.metric)

            # a list of results if args.bootstrap, a single prediction otherwise
            random_selection_perf, info_selection_perf = evaluator.get_per_sampling_performance(model, tokenizer,
                                                                                                args.use_cache)
            if not args.bootstrap:
                # unify the format, so we have a single result formatting
                random_performance_to_print, info_performance_to_print = [random_selection_perf], [info_selection_perf]

            elif args.aggregate_results:
                # bootstrapping + aggregation
                mean = sum(random_selection_perf) / len(random_selection_perf)
                q_lower = np.quantile(random_selection_perf, q=0.025)
                q_upper = np.quantile(random_selection_perf, q=0.975)
                broader_q = max((math.fabs(mean - q_lower), math.fabs(mean - q_upper)))

                random_performance_to_print = ["{:.5f}±{:.5f}".format(mean, broader_q)]

                mean = sum(info_selection_perf) / len(info_selection_perf)
                q_lower = np.quantile(info_selection_perf, q=0.025)
                q_upper = np.quantile(info_selection_perf, q=0.975)
                broader_q = max((math.fabs(mean - q_lower), math.fabs(mean - q_upper)))

                info_performance_to_print = ["{:.5f}±{:.5f}".format(mean, broader_q)]

            else:
                # bootstrapping, but reporting of all results
                # no aggregation, we just round up the floats for printing
                random_performance_to_print = ["{:.5f}".format(perf_one) for perf_one in random_selection_perf]
                info_performance_to_print = ["{:.5f}".format(perf_one) for perf_one in info_selection_perf]

            for random_perf_one, info_perf_one in zip(random_performance_to_print, info_performance_to_print):
                print("{}\t{}\t{}\t{}\t{}".format(model_name_or_path,
                                                  dataset_id,
                                                  template_id,
                                                  random_perf_one,
                                                  info_perf_one))

            result_key = "%s-%s" % (template_id, template_id)
            results[model_name_or_path][result_key] = {"random": random_performance_to_print,
                                                       "info": info_performance_to_print}

        pd.DataFrame(results[model_name_or_path]).to_csv("%s_sensitivity_evaluation.tsv"
                                                         % model_name_or_path.split("/")[-1], sep="\t")
