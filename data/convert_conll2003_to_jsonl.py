import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Union


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))

    basename = os.path.basename(args.input)
    with open(args.output, mode="w", encoding="utf-8") as f:
        for i, sentences in enumerate(read_conll(args.input)):
            document_id = f"{basename}-{i}"
            for j, sentence in enumerate(sentences):
                sentence["id"] = f"{document_id}-{j}"
            document = {"id": document_id, "examples": sentences}
            f.write(encoder.encode(document))
            f.write("\n")


def read_conll(file: Union[str, bytes, os.PathLike]) -> Iterable[List[Dict[str, Any]]]:
    sentences: List[Dict[str, Any]] = []
    words: List[str] = []
    labels: List[str] = []

    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART"):
                if sentences:
                    yield sentences
                    sentences = []
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels))
                    words = []
                    labels = []
            else:
                cols = line.split(" ")
                words.append(cols[0])
                labels.append(cols[-1])

    if sentences:
        yield sentences


def _conll_to_example(words: List[str], tags: List[str]) -> Dict[str, Any]:
    text, positions = _conll_words_to_text(words)
    entities = [
        {"start": positions[start][0], "end": positions[end - 1][1], "label": label}
        for start, end, label in _conll_tags_to_spans(tags)
    ]
    return {"text": text, "entities": entities, "word_positions": positions}


def _conll_words_to_text(words: Iterable[str]) -> Tuple[str, List[Tuple[int, int]]]:
    text = ""
    positions = []
    offset = 0
    for word in words:
        if text:
            text += " "
            offset += 1
        text += word
        n = len(word)
        positions.append((offset, offset + n))
        offset += n
    return text, positions


def _conll_tags_to_spans(tags: Iterable[str]) -> Iterable[Tuple[int, int, str]]:
    # NOTE: assume IO scheme
    start, label = -1, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            if start >= 0:
                assert label is not None
                yield (start, i, label)
                start, label = -1, None
        else:
            cur_label = tag[2:]
            if cur_label != label:
                if start >= 0:
                    assert label is not None
                    yield (start, i, label)
                start, label = i, cur_label


if __name__ == "__main__":
    main()
