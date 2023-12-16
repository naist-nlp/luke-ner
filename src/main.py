import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    LukeForEntitySpanClassification,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import Collator, Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    model: str = "studio-ousia/mluke-base-lite"
    cache_dir: Optional[str] = None
    max_entity_length: int = 128
    max_mention_length: int = 16
    pretokenize: bool = False


def main(args: Arguments, training_args: TrainingArguments):
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"args: {args}")
    logger.info(f"training args: {training_args}")

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    cache_dir = args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    config = AutoConfig.from_pretrained(args.model)
    if config.label2id == PretrainedConfig().label2id:
        if "train" not in raw_datasets:
            raise RuntimeError("Cannot retrieve labels from dataset")
        label_set = set()
        for document in raw_datasets["train"]:
            for example in document["examples"]:
                for entity in example["entities"]:
                    label_set.add(entity["label"])
        labels = ["<none>"] + sorted(label_set)
        config.label2id = {label: i for i, label in enumerate(labels)}
        config.id2label = {i: label for i, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        task="entity_span_classification",
        max_entity_length=args.max_entity_length,
        max_mention_length=args.max_mention_length,
    )

    preprocessor = Preprocessor(
        tokenizer,
        labels=[v for _, v in sorted(config.id2label.items())],
        extend_context=True,
        split_entity_spans=True,
        max_entity_length=args.max_entity_length,
        max_mention_length=args.max_mention_length,
        pretokenize=args.pretokenize,
    )
    # print(f"LABELS: {preprocessor.label2id}")

    def preprocess(documents):
        features = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    set_seed(training_args.seed)

    model = LukeForEntitySpanClassification.from_pretrained(args.model, config=config)

    trainer = SpanClassificationTraier(
        model=model,
        args=training_args,
        train_dataset=splits.get("train"),
        eval_dataset=splits.get("validation"),
        data_collator=Collator(tokenizer),
    )

    # Add logger callback

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()

        logits = trainer.last_prediction.predictions
        predictions = predict(logits, splits["validation"], config.id2label)
        new_metrics = evaluate(predictions, raw_datasets["validation"])
        metrics.update({f"eval_exact_{k}": v for k, v in new_metrics.items()})

        logger.info(f"eval metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        result = trainer.predict(splits["test"])

        logits = trainer.last_prediction.predictions
        predictions = predict(logits, splits["test"], config.id2label)
        new_metrics = evaluate(predictions, raw_datasets["test"])
        result.metrics.update({f"test_exact_{k}": v for k, v in new_metrics.items()})

        logger.info(f"test metrics: {result.metrics}")
        trainer.log_metrics("predict", result.metrics)
        if training_args.save_strategy != "no":
            trainer.save_metrics("predict", result.metrics)


class SpanClassificationTraier(Trainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("compute_metrics", self._compute_metrics)
        super().__init__(*args, **kwargs)

    def _compute_metrics(self, p: EvalPrediction):
        self.last_prediction = p
        return _compute_metrics(p)


def _compute_metrics(p: EvalPrediction):
    # NOTE: This is not an accurate calculation of recall because some gold entities may be discarded during preprocessing.
    preds = p.predictions.argmax(axis=-1).ravel()
    labels = p.label_ids.ravel()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    pred_entity_mask = preds != 0
    gold_entity_mask = labels != 0
    num_corrects = (preds[gold_entity_mask] == labels[gold_entity_mask]).sum().item()
    num_preds = pred_entity_mask.sum().item()
    num_golds = gold_entity_mask.sum().item()
    precision = num_corrects / num_preds if num_preds > 0 else float("nan")
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(predictions, dataset):
    pred_entities = predictions
    true_entities = OrderedDict()

    for document in dataset:
        for example in document["examples"]:
            entities = set((ent["start"], ent["end"], ent["label"]) for ent in example["entities"])
            true_entities[example["id"]] = entities

    assert len(pred_entities) == len(true_entities)
    num_corrects = sum(len(y & t) for y, t in zip(pred_entities.values(), true_entities.values()))
    num_preds = sum(len(y) for y in pred_entities.values())
    num_golds = sum(len(t) for t in true_entities.values())
    precision = num_corrects / num_preds if num_preds > 0 else float("nan")
    recall = num_corrects / num_golds if num_golds > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")

    return {"precision": precision, "recall": recall, "f1": f1}


def predict(logits, features, id2label):
    outputs = OrderedDict()
    preds = logits.argmax(axis=-1)
    assert len(preds) == len(features)
    for i, f in enumerate(features):
        if f["id"] not in outputs:
            outputs[f["id"]] = set()
        entities = outputs[f["id"]]
        char_spans = f["char_spans"]
        for idx, (start, end) in zip(preds[i, : len(char_spans)], char_spans):
            if idx != 0:
                entities.add((start, end, id2label[idx]))
    return outputs


def setup_logger(training_args: TrainingArguments):
    import sys, datasets, transformers  # isort:skip  # noqa

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if training_args.should_log and training_args.save_strategy != "no":
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "main.log"))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_propagation()
    getattr(transformers.utils.logging, "disable_default_handler", lambda: None)()
    datasets.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.enable_propagation()
    getattr(datasets.utils.logging, "disable_default_handler", lambda: None)()
    logger.setLevel(log_level)


if __name__ == "__main__":
    CONFIG_FILE = Path(__file__).parents[1] / "default.conf"
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args_filename=CONFIG_FILE)
    if args.validation_file is None:
        training_args.evaluation_strategy = "no"
    main(args, training_args)
