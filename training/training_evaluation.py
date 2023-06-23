from typing import List, Optional

import torch
from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import Head, AdaptationDataset
from transformers import PreTrainedTokenizer

from evaluations.qa_evaluator import QAEvaluator


class F1ForQA(QAEvaluator, EvaluatorBase):
    compatible_heads: List[Head] = [Head.SEQ2SEQ, Head.QA]

    # noinspection PyMissingConstructor
    def __init__(self,
                 eval_data_path: str,
                 prompt: str,
                 model_type: str,
                 answer_type: Optional[str] = None,
                 prompt_lang: str = "cs",
                 num_probed_examples: int = 3,
                 verbose: bool = False,
                 firstn: int = 123,
                 device: Optional[str] = None,
                 decides_convergence: bool = False):
        # prompting:
        self.answer_type = answer_type
        self.prompt = prompt
        self.prompt_lang = prompt_lang
        # priming:
        self.num_primed_examples = num_probed_examples

        # boring:
        self.eval_data_path = eval_data_path
        self.verbose = verbose
        self.firstn = firstn
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.determines_convergence = decides_convergence
        self.model_type = model_type

    def __call__(self,
                 model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> float:
        self.model = model
        self.tokenizer = tokenizer

        if self.num_primed_examples > 0:
            _, _, fscore = self.evaluate_priming(self.num_primed_examples, self.prompt,
                                                 self.answer_type, self.verbose, self.firstn)
        else:
            _, _, fscore = self.evaluate_prompt(self.prompt, self.answer_type, self.verbose, self.firstn)

        return fscore

    def __str__(self) -> str:
        return "%s-%s" % (self.answer_type, super().__str__())
