from typing import List, Dict

import pandas as pd

from evaluation.tasks.task import Task

# difference of training and validation concept sets
unseen_concepts = {'select filter count filter count arithmetic_subtraction',
                   'select project grouped_count kth_highest',
                   'select project grouped_count filter_a_where_b_is_given_value count',
                   'select select select union count',
                   'select project grouped_count maximum_number minimum_number arithmetic_subtraction',
                   'select maximum_number list_subtraction maximum_number arithmetic_sum_multiple',
                   'select maximum_number kth_highest kth_highest arithmetic_sum_multiple arithmetic_subtraction',
                   'select project grouped_count kth_highest kth_lowest arithmetic_subtraction',
                   'select project filter filter list_subtraction count',
                   'select filter filter filter project arithmetic_subtraction',
                   'select project project union project arithmetic_mean_single',
                   'select project project filter_a_where_b_is_min project count',
                   'select filter project project arithmetic_subtraction',
                   'select maximum_number list_subtraction arithmetic_sum_single arithmetic_subtraction'}

teabreac_template = "Question: %s Context: %s Answer:"


class TeaBReAC(Task):

    def __init__(self, teabreac_val_dataset_jsonl: str):
        super().__init__(".")

        tea_val = pd.read_json(teabreac_val_dataset_jsonl, lines=True)
        tea_val["context_text"] = tea_val["context_text"].apply(lambda c: c.replace(" -> ", ". "))
        tea_val["answers_text"] = tea_val["answers_objects"].apply(lambda ans_obj: self._get_answer(ans_obj))
        tea_val["program_modules_str"] = tea_val["program_modules"].apply(lambda modules: self._get_categories(modules))

        unseen_val = tea_val[tea_val["program_modules_str"].isin(unseen_concepts)]

        prompts = unseen_val.apply(lambda row: teabreac_template % (row["question_text"], row["context_text"]), axis=1)

        self.data = list(zip(prompts, unseen_val["answers_text"], unseen_val["program_modules_str"]))

    @staticmethod
    def _get_answer(ans_object: List[Dict[str, str]]) -> str:
        # list([{'number': '121', 'date': {'day': '', 'month': '', 'year': ''}, 'spans': []}])
        return " ".join(ans['number'] if ans['number']
                        else " ".join(ans['date'].values()) if " ".join(ans['date'].values())
                        else " ".join(ans['spans']) for ans in ans_object)

    @staticmethod
    def _get_categories(program_modules: List[str]) -> str:
        return " ".join(program_modules)
