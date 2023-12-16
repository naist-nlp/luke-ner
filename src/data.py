import itertools
import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import torch
from transformers import (
    DataCollatorWithPadding,
    LukeTokenizer,
    MLukeTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


class Entity(TypedDict):
    start: int
    end: int
    label: str


class Example(TypedDict):
    id: str
    text: str
    entities: List[Entity]
    word_positions: Optional[List[Tuple[int, int]]]


class Preprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        labels: List[str],
        non_entity_label: Optional[str] = None,
        extend_context: bool = True,
        split_entity_spans: bool = False,
        max_sequence_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        max_mention_length: Optional[int] = None,
        pretokenize: bool = False,
    ):
        if isinstance(tokenizer, (LukeTokenizer, MLukeTokenizer)):
            import tokenization_utils  # noqa
        else:
            raise RuntimeError(
                "Only `LukeTokenizer` and `MLukeTokenizer` are currently supported,"
                f" but got `{type(tokenizer).__name__}`."
            )

        self.tokenizer = tokenizer
        if tokenizer.is_fast:
            self._fast_tokenizer = tokenizer
        else:
            self._fast_tokenizer = PreTrainedTokenizerFast(__slow_tokenizer=tokenizer)

        if not non_entity_label:
            non_entity_label = labels[0]
        elif non_entity_label not in labels:
            raise ValueError(f"{non_entity_label=} is not included in labels.")
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.non_entity_label = non_entity_label

        self.extend_context = extend_context
        self.split_entity_spans = split_entity_spans

        if max_sequence_length is None:
            max_sequence_length = tokenizer.model_max_length
        num_specials = len(tokenizer.build_inputs_with_special_tokens([]))
        self.max_sequence_length = max_sequence_length
        self.max_num_tokens = max_sequence_length - num_specials

        tokenizer_max_entity_length = getattr(tokenizer, "max_entity_length", None)
        if max_entity_length is None:
            max_entity_length = tokenizer_max_entity_length
        elif max_entity_length != tokenizer_max_entity_length:
            logger.warning(
                "`preprocessor.max_entity_length` is different from the tokenizer's:"
                f" ({max_entity_length} != {tokenizer_max_entity_length})"
            )
        self.max_entity_length = max_entity_length

        tokenizer_max_mention_length = getattr(tokenizer, "max_mention_length", None)
        if max_mention_length is None:
            max_mention_length = tokenizer_max_mention_length
        elif max_mention_length != tokenizer_max_mention_length:
            logger.warning(
                "`preprocessor.max_mention_length` is different from the tokenizer's:"
                f" ({max_mention_length} != {tokenizer_max_mention_length})"
            )
        self.max_mention_length = max_mention_length

        if self.split_entity_spans and self.max_entity_length is None:
            raise ValueError(
                "`max_entity_length` must be specified when `split_entity_spans` is True."
            )

        self.pretokenize = pretokenize

    def __call__(self, document: List[Example]) -> Iterable[Dict[str, Any]]:
        # NOTE: Whereas the original implementation tokenizes text word by word, this tokenizes raw text at one time.
        # https://github.com/studio-ousia/luke/blob/2b066b8df0bb1dae8812ef40733b6d0194517a29/examples/ner/reader.py#L107
        #
        # Because tokenization is done on raw text in LUKE pre-training,
        # word-by-word tokenization may result in different tokenization from pre-training.
        # Word segmentation should be optional for enumerating word-based spans, as it is not a prerequisite for tokenization.

        segments = None
        if self.pretokenize:
            segments = []
            for example in document:
                if example["word_positions"] is None:
                    raise ValueError("`word_positions` is required for pretokenization.")
                segments.append(example["word_positions"])

        for example, tokenization in zip(
            document, self.tokenize([e["text"] for e in document], segments)
        ):
            entity_map = {(ent["start"], ent["end"]): ent["label"] for ent in example["entities"]}

            for token_spans, char_spans in self._batch_spans(example, tokenization):
                # Currently, this works only with `LukeTokenizer` or `MLukeTokenizer`.
                assert isinstance(self.tokenizer, (LukeTokenizer, MLukeTokenizer))
                encoding = self.tokenizer.prepare_for_model(
                    tokenization["token_ids"],
                    entity_ids=[self.tokenizer.entity_mask_token_id] * len(token_spans),
                    entity_token_spans=token_spans,
                    add_special_tokens=True,
                )
                encoding["labels"] = [
                    self.label2id[entity_map.pop((start, end), self.non_entity_label)]
                    for start, end in char_spans
                ]
                encoding["char_spans"] = char_spans
                encoding["id"] = example["id"]
                yield encoding

            if entity_map:
                logger.warning(f"Some entities are discarded during preprocessing: {entity_map}")

    def _batch_spans(self, example, tokenization):
        def _enumerate_spans(left, right, max_length):
            if max_length is None:
                max_length = right - left
            for i in range(left, right):
                for j in range(i + 1, right + 1):
                    if j - i > max_length:
                        continue
                    yield i, j

        token_spans = []
        char_spans = []

        text = example["text"]
        offsets = tokenization["offsets"]
        boundaries = set(itertools.chain.from_iterable(example["word_positions"] or []))
        seq_start, seq_end = tokenization["context_boundary"]
        for start, end in _enumerate_spans(seq_start, seq_end, self.max_mention_length):
            char_start, char_end = offsets[start - seq_start][0], offsets[end - seq_start - 1][1]
            if text[char_start] == " ":
                char_start += 1
            if char_start == offsets[start - seq_start][1]:
                continue
            assert char_start < char_end
            if boundaries and not (char_start in boundaries and char_end in boundaries):
                continue
            token_spans.append((start, end))
            char_spans.append((char_start, char_end))
        assert len(char_spans) == len(set(char_spans))

        if self.split_entity_spans:
            assert self.max_entity_length is not None
            for i in range(0, len(token_spans), self.max_entity_length):
                yield (
                    token_spans[i : i + self.max_entity_length],
                    char_spans[i : i + self.max_entity_length],
                )
        else:
            num_spans = len(token_spans)
            if self.max_entity_length is not None and num_spans > self.max_entity_length:
                logger.info(f"Truncate spans: {num_spans} -> {self.max_entity_length}")
                num_spans = self.max_entity_length
            yield token_spans[:num_spans], char_spans[:num_spans]

    def tokenize(
        self, document: List[str], segments: Optional[List[List[Tuple[int, int]]]] = None
    ) -> Iterable[Dict[str, Any]]:
        if segments is not None:
            assert len(document) == len(segments)
            batch_text = [
                text[start:end]
                for text, positions in zip(document, segments)
                for start, end in positions
            ]
        else:
            batch_text = document

        encoding = self._fast_tokenizer(
            batch_text,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        _post_process_encoding(encoding, batch_text, self._fast_tokenizer)
        encoding = _merge_encoding(encoding, segments, self._fast_tokenizer)
        all_tokens = [token for tokens in encoding["input_ids"] for token in tokens]

        offset = 0
        for i, tokens in enumerate(encoding["input_ids"]):
            n = len(tokens)
            if n > self.max_num_tokens:
                logger.info(f"truncate sequence: {encoding['input_ids'][i]}")
                tokens = tokens[: self.max_num_tokens]
                boundary = (0, self.max_num_tokens)
            elif self.extend_context:
                start, end = offset, offset + n
                left_length, right_length = start, len(all_tokens) - end
                m = self.max_num_tokens - n
                if left_length < right_length:
                    left_context_length = min(left_length, m // 2)
                    right_context_length = min(right_length, m - left_context_length)
                else:
                    right_context_length = min(right_length, m // 2)
                    left_context_length = min(left_length, m - right_context_length)
                tokens = all_tokens[start - left_context_length : end + right_context_length]
                boundary = (left_context_length, left_context_length + n)
            else:
                boundary = (0, n)

            yield {
                "token_ids": tokens,
                "context_boundary": boundary,
                "offsets": encoding["offset_mapping"][i],
            }
            offset += n


def _post_process_encoding(encoding, batch_text, tokenizer):
    prefix_token_id = tokenizer.convert_tokens_to_ids("▁")
    if prefix_token_id == tokenizer.unk_token_id:
        return
    if not getattr(tokenizer, "add_prefix_space", True):
        return

    for i, text in enumerate(batch_text):
        if len(encoding["input_ids"][i]) == 0:
            # some tokenizers return empty ids for illegal characters.
            continue
        if encoding["input_ids"][i][0] != prefix_token_id or text[0].isspace():
            continue
        # set to (0, 0) because there is no corresponding character.
        encoding["offset_mapping"][i][0] = (0, 0)


def _merge_encoding(encoding, segments, tokenizer):
    if not segments:
        return encoding

    new_encoding = {"input_ids": [], "offset_mapping": []}

    add_prefix_space = getattr(tokenizer, "add_prefix_space", True)
    index = 0
    for positions in segments:
        merged_input_ids = []
        merged_offsets = []

        batch_input_ids = encoding["input_ids"][index : index + len(positions)]
        batch_offsets = encoding["offset_mapping"][index : index + len(positions)]

        for i, (input_ids, offsets, (char_start, _)) in enumerate(
            zip(batch_input_ids, batch_offsets, positions)
        ):
            if not input_ids:  # skip illegal input_ids
                continue
            if add_prefix_space and i > 0 and offsets[0] == (0, 0):
                input_ids = input_ids[1:]
                offsets = offsets[1:]
            merged_input_ids.extend(input_ids)
            merged_offsets.extend((char_start + ofs[0], char_start + ofs[1]) for ofs in offsets)

        assert all(
            merged_offsets[i][0] >= merged_offsets[i - 1][1] for i in range(1, len(merged_offsets))
        )

        new_encoding["input_ids"].append(merged_input_ids)
        new_encoding["offset_mapping"].append(merged_offsets)
        index += len(positions)
    assert index == len(encoding["input_ids"])

    return new_encoding


class Collator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f.copy() for f in features]
        labels = [f.pop("labels") for f in features]
        extra_fields = {}
        extra_field_names = {"id", "char_spans"}
        for k in list(features[0].keys()):
            if k in extra_field_names:
                extra_fields[k] = [f.pop(k) for f in features]

        batch = super().__call__(features)
        batch.update(extra_fields)

        entity_length = batch["entity_ids"].shape[1]
        if self.tokenizer.padding_side == "right":
            batch["labels"] = [label + [-100] * (entity_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[-100] * (entity_length - len(label)) + label for label in labels]

        if self.return_tensors == "pt":
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        else:
            warnings.warn(f"return_tensors='{self.return_tensors}' is not supported.")

        return batch
