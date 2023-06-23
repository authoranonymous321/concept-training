from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from evaluation.tasks.en.superglue import Broadcoverage, BoolQ, WinogradSchema, CoPA, MultiRC, CommitmentBank, RTE, WiC, \
    ReCoRD, Winogender
from evaluator import Evaluator

model_path = "gaussalgo/mt5-base-priming-QA_en-cs"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

all_tasks = [
    Broadcoverage(),
    BoolQ(),
    CommitmentBank(),
    WinogradSchema(),
    CoPA(),
    MultiRC(),
    RTE(),
    WiC(),
    ReCoRD(),
    Winogender()
]

evaluator = Evaluator()
evaluations = evaluator.evaluate(model, tokenizer, all_tasks)

print("Evaluation done: %s" % evaluations)
print()
