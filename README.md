# LUKE-NER

This repository is for NER training/inference using [LUKE](https://github.com/studio-ousia/luke).

Features:
- Our implementation relies on `Trainer` of [huggingface/transformers](https://github.com/huggingface/transformers) for easier use (while the official repository provides examples using [AllenNLP](https://github.com/allenai/allennlp)).
- This repository improves preprocessing for non-space-delimited languages.
- The code is compatible with pre-trained LUKE models available on Hugging Face Hub.

## Usage

### Fine-tuning

```py
.venv/bin/torchrun --nproc_per_node 4 src/main.py \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file data/eng.train.jsonl \
    --validation_file data/eng.testa.jsonl \
    --test_file data/eng.testb.jsonl \
    --model "studio-ousia/luke-large-lite" \
    --output_dir ./output/ \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --max_entity_length 64 \
    --max_mention_length 16 \
    --save_strategy epoch
```

### Evaluation/Prediction

```py
.venv/bin/torchrun --nproc_per_node 4 src/main.py \
    --do_eval \
    --do_predict \
    --validation_file data/eng.testa.jsonl \
    --test_file data/eng.testb.jsonl \
    --model PATH_TO_YOUR_MODEL \
    --output_dir ./output/ \
    --per_device_eval_batch_size 8 \
    --max_entity_length 64 \
    --max_mention_length 16
```

## Performances

### CoNLL '03 English (test)

| Model | Precision | Recall | F1 |
| --- | :---: | :---: | :---: |
| LUKE ([paper](https://aclanthology.org/2020.emnlp-main.523/)) | - | - | 94.3 |
| [studio-ousia/luke-large-finetuned-conll-2003](https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003) on [notebook](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb) | 93.86 | 94.53 | 94.20 |
| [studio-ousia/luke-large-finetuned-conll-2003](https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003) on [script](https://github.com/studio-ousia/luke/blob/master/examples/ner/evaluate_transformers_checkpoint.py) | 94.58 | 94.65 | 94.61 |
| [studio-ousia/luke-large-finetuned-conll-2003](https://huggingface.co/studio-ousia/luke-large-finetuned-conll-2003) on our code | 93.98 | 94.67 | 94.33 |
| [studio-ousia/luke-large-lite](https://huggingface.co/studio-ousia/luke-large-lite) fine-tuned with our code | N/A | N/A | N/A |
| mLUKE ([paper](https://aclanthology.org/2022.acl-long.505/)) | - | - | 94.0 |
| [studio-ousia/mluke-large-lite-finetuned-conll-2003](https://huggingface.co/studio-ousia/mluke-large-lite-finetuned-conll-2003) on [notebook](https://colab.research.google.com/github/studio-ousia/luke/blob/master/notebooks/huggingface_conll_2003.ipynb)* | 94.23 | 94.23 | 94.23 |
| [studio-ousia/mluke-large-lite-finetuned-conll-2003](https://huggingface.co/studio-ousia/mluke-large-lite-finetuned-conll-2003) on [script](https://github.com/studio-ousia/luke/blob/master/examples/ner/evaluate_transformers_checkpoint.py)* | 94.33 | 93.76 | 94.05 |
| [studio-ousia/mluke-large-lite-finetuned-conll-2003](https://huggingface.co/studio-ousia/mluke-large-lite-finetuned-conll-2003) on our code* | 93.76 | 93.92 | 93.84 |
| [studio-ousia/mluke-large-lite](https://huggingface.co/studio-ousia/mluke-large-lite) fine-tuned with our code | N/A | N/A | N/A |

Note that the codes marked with `*` are a bit tweaked when evaluating [studio-ousia/mluke-large-lite-finetuned-conll-2003](https://huggingface.co/studio-ousia/mluke-large-lite-finetuned-conll-2003) because the current model was fine-tuned with erroneous `entity_attention_mask` (See the issues [#166](https://github.com/studio-ousia/luke/issues/166#issuecomment-1578524458), [#172](https://github.com/studio-ousia/luke/pull/172) for details).
