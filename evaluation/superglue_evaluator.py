import argparse
import math

import numpy as np
import pandas as pd
import torch
from promptsource.templates import DatasetTemplates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.evaluator_informative import RougeInformative, \
    AccuracyInformative
from evaluation.tasks.en.superglue import all_task_classes

parser = argparse.ArgumentParser()

parser.add_argument("--model_names_or_paths", type=str,
                    help="Coma-separated list of evaluated models' identifiers")
parser.add_argument("--use_cache", type=str, default="False", choices=('True', 'False'),
                    help="Whether to use cached predictions, if available.")
parser.add_argument("--firstn", type=int, default=1000,
                    help="If given, a number of samples from dataset to evaluate on.")
parser.add_argument("--metric", default="ROUGE", type=str,
                    help="A metric to compute informative difference with. Must be one of the implemented metrics:"
                         "'ROUGE', 'Accuracy'.")
parser.add_argument("--max_input_length", default=None, type=int,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--bootstrap", default="False", type=str,
                    help="Whether to collect a set of results over random subsets of predictions. Defaults to True.")
parser.add_argument("--tasks", default="axb,boolq,cb,wsc,copa,multirc,rte,wic,record,axg", type=str,
                    help="Coma-separated list of SuperGLUE tasks' ids. See default values for selection.")

args = parser.parse_args()

args.use_cache = args.use_cache != "False"
args.bootstrap = args.bootstrap != "False"

results = {}

# eval iteration
for model_name_or_path in args.model_names_or_paths.split(","):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    results[model_name_or_path] = {}

    selected_tasks_ids = args.tasks.split(",")
    selected_tasks_classes = [SCls for SCls in all_task_classes
                              if any(t_id in SCls.promptsource_id for t_id in selected_tasks_ids)]
    for SGLUETaskClass in selected_tasks_classes:
        for template_name in DatasetTemplates(SGLUETaskClass.promptsource_id).all_template_names:
            task = SGLUETaskClass(prompts_template=template_name)

            if args.metric == "ROUGE":
                evaluator = RougeInformative(task,
                                             bootstrap=args.bootstrap,
                                             max_input_length=args.max_input_length,
                                             firstn=args.firstn if args.firstn else None)
            elif args.metric == "Accuracy":
                evaluator = AccuracyInformative(task, bootstrap=args.bootstrap, max_input_length=args.max_input_length)
            else:
                raise ValueError("Unknown metric: %s" % args.metric)

            # a list of results if args.bootstrap, a single prediction otherwise
            random_selection_perf, info_selection_perf = evaluator.get_per_sampling_performance(model, tokenizer,
                                                                                                args.use_cache)
            if not args.bootstrap:
                # unify the format, so we have a single result formatting
                random_performance_to_print, info_performance_to_print = random_selection_perf, info_selection_perf

            else:
                mean = sum(random_selection_perf) / len(random_selection_perf)
                q_lower = np.quantile(random_selection_perf, q=0.025)
                q_upper = np.quantile(random_selection_perf, q=0.975)
                broader_q = max((math.fabs(mean - q_lower), math.fabs(mean - q_upper)))

                random_performance_to_print = "{:.5f}±{:.5f}".format(mean, broader_q)

                mean = sum(info_selection_perf) / len(info_selection_perf)
                q_lower = np.quantile(info_selection_perf, q=0.025)
                q_upper = np.quantile(info_selection_perf, q=0.975)
                broader_q = max((math.fabs(mean - q_lower), math.fabs(mean - q_upper)))

                info_performance_to_print = "{:.5f}±{:.5f}".format(mean, broader_q)

            print("{}\t{}\t{}\t{}\t{}".format(model_name_or_path,
                                              task.promptsource_id,
                                              template_name,
                                              random_performance_to_print,
                                              info_performance_to_print))

            result_key = "%s-%s" % (task.promptsource_id, template_name)
            results[model_name_or_path][result_key] = {"random": random_performance_to_print,
                                                       "info": info_performance_to_print}

    pd.DataFrame(results[model_name_or_path]).to_csv("%s_superglue_evaluation.tsv"
                                                     % model_name_or_path.split("/")[-1], sep="\t")
