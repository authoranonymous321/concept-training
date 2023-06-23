import abc
import copy
import random
from typing import Optional, Tuple, List, Union

from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.evaluators.generative import ROUGE
from transformers import PreTrainedTokenizer, PreTrainedModel

from evaluation.evaluator import Evaluator
from evaluation.tasks.task import Task

random.seed(42)


class InformativeEvaluatorBase:

    def __init__(self,
                 task: Task,
                 num_demonstrations: int = 3,
                 firstn: Optional[int] = None,
                 bootstrap: bool = False,
                 max_input_length: Optional[int] = None,
                 reuse_last_run: bool = False,
                 randomize_labels: bool = False,
                 flip_labels: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_demonstrations = num_demonstrations
        self.firstn = firstn
        self.bootstrap = bootstrap
        self.max_input_length = max_input_length
        self.reuse_last_run = reuse_last_run
        self.randomize_labels = randomize_labels
        self.flip_labels = flip_labels

    @abc.abstractmethod
    def _compute(self, expected: List[str], actual: List[str]) -> float:
        pass

    def _randomize_labels(self, task: Task) -> Task:
        dummy_labels = ["Foo", "Bar", "Tic", "Woe"]
        all_labels = set(sample[1] for sample in task.data)
        assert len(all_labels) <= len(dummy_labels), "We got too many distinct labels in the task %s" % task

        all_labels_mapping = {orig_label: dummy_labels[i] for i, orig_label in enumerate(all_labels)}
        task.data = [(sample[0], all_labels_mapping[sample[1]], sample[2]) for sample in task.data]

        # replace labels as well as in the demonstrations' prompts
        for orig_label, dummy_label in all_labels_mapping.items():
            task.data = [(sample[0].replace(orig_label, dummy_label).replace(orig_label.lower(), dummy_label),
                          sample[1], sample[2]) for sample in task.data]

        return task

    def _flip_labels(self, task: Task) -> Task:
        all_labels = set(sample[1] for sample in task.data)
        swapped_labels = random.sample(all_labels, k=len(all_labels))

        all_labels_mapping = dict(zip(all_labels, swapped_labels))
        task.data = [(sample[0], all_labels_mapping[sample[1]], sample[2]) for sample in task.data]
        return task

    def _compute_bootstrapped(self,
                              expected_all: List[str],
                              actual_all: List[str],
                              per_round_samples: int = 100,
                              repeats: int = 200) -> List[float]:
        assert len(expected_all) == len(actual_all), "Prediction lists' length do not match"

        evals = []
        while len(evals) < repeats:
            subset_idx = [random.randrange(len(expected_all)) for _ in range(per_round_samples)]
            expected_subset = [expected_all[idx] for idx in subset_idx]
            actual_subset = [actual_all[idx] for idx in subset_idx]

            evals.append(self._compute(expected_subset, actual_subset))

        return evals

    def get_per_sampling_performance(self,
                                     model: PreTrainedModel,
                                     tokenizer: PreTrainedTokenizer,
                                     use_cache: bool = True) -> Tuple[Union[List[float], float],
                                                                      Union[List[float], float]]:
        # there's always less samples in 'informative' group, so we start with that
        if self.randomize_labels:
            info_task = self._randomize_labels(copy.deepcopy(self.task))
            selection_strategy = "random"
        elif self.flip_labels:
            info_task = self._flip_labels(copy.deepcopy(self.task))
            selection_strategy = "random"
        else:
            info_task = self.task
            selection_strategy = "clustered"

        expected_info, actual_info, eval_set = Evaluator.collect_predictions(model, tokenizer, info_task,
                                                                             self.num_demonstrations, self.firstn,
                                                                             demo_selection_strategy=selection_strategy,
                                                                             max_input_length=self.max_input_length,
                                                                             use_cache=use_cache)
        expected, actual_random, _ = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="random",
                                                                   eval_set=None
                                                                   if self.randomize_labels or self.flip_labels
                                                                   else eval_set,
                                                                   use_cache=use_cache)
        if self.bootstrap:
            informative_performance = self._compute_bootstrapped(expected_info, actual_info)
            random_performance = self._compute_bootstrapped(expected, actual_random)
        else:
            informative_performance = self._compute(expected_info, actual_info)
            random_performance = self._compute(expected, actual_random)

        # print("Model's performance in informative selection: %s" % informative_performance)

        return random_performance, informative_performance

    def __str__(self):
        return "%s_%s" % (self.task, super().__str__())


class RougeRandom(InformativeEvaluatorBase, ROUGE):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        return self.evaluate_str(expected, actual)

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="random",
                                                                   max_input_length=self.max_input_length,
                                                                   use_cache=self.reuse_last_run)
        random_performance = self._compute(expected, actual)

        return random_performance


class RougeInformative(RougeRandom, ROUGE):

    def __call__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, _) -> float:
        expected, actual, eval_set = Evaluator.collect_predictions(model, tokenizer, self.task,
                                                                   self.num_demonstrations, self.firstn,
                                                                   demo_selection_strategy="clustered",
                                                                   max_input_length=self.max_input_length,
                                                                   use_cache=self.reuse_last_run)
        info_performance = self._compute(expected, actual)

        return info_performance


class AccuracyRandom(RougeRandom, EvaluatorBase):

    def _compute(self, expected: List[str], actual: List[str]) -> Union[float, List[float]]:
        num_correct = sum([exp == act for exp, act in zip(expected, actual)])
        return num_correct / len(expected)


class AccuracyInformative(AccuracyRandom, RougeInformative, EvaluatorBase):
    pass
