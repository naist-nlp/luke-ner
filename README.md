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

TODO
