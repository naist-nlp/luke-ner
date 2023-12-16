from transformers.convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS,
    RobertaConverter,
    XLMRobertaConverter,
)


class LukeConverter(RobertaConverter):
    pass


class MLukeConverter(XLMRobertaConverter):
    pass


SLOW_TO_FAST_CONVERTERS.setdefault("LukeTokenizer", LukeConverter)
SLOW_TO_FAST_CONVERTERS.setdefault("MLukeTokenizer", MLukeConverter)
