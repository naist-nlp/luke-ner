import logging
import logging.config
import time
import typing
from enum import Enum
from pathlib import Path

import datasets
import tqdm
import transformers


class LoggerCallback(transformers.TrainerCallback):
    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            _ = logs.pop("total_flos", None)
            self.logger.info(f"logs at {state.global_step}: {logs}")


def setup_logger(training_args: transformers.TrainingArguments):
    _configure_logging(training_args)
    logging.captureWarnings(True)

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_propagation()
    getattr(transformers.utils.logging, "disable_default_handler", lambda: None)()
    datasets.utils.logging.set_verbosity(log_level)
    datasets.utils.logging.enable_propagation()
    getattr(datasets.utils.logging, "disable_default_handler", lambda: None)()
    logging.getLogger("__main__").setLevel(log_level)


def _configure_logging(training_args: transformers.TrainingArguments):
    base_format = "%(asctime)s - %(levelname)s (%(name)s) - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S.%f %Z"

    config: typing.Dict[str, typing.Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": Formatter,
                "format": base_format,
                "datefmt": date_format,
            },
            "color": {
                "()": ColoredFormatter,
                "format": base_format,
                "datefmt": date_format,
            },
        },
        "handlers": {
            "stream": {
                "()": _TqdmLoggingHandler,
                "formatter": "color",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "WARNING",
            "handlers": ["stream"],
        },
    }

    if training_args.do_train and training_args.should_log and training_args.save_strategy != "no":
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": output_dir / "training.log",
        }
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)


# adaped from `https://github.com/tqdm/tqdm/blob/v4.65.0/tqdm/contrib/logging.py#L16`
class _TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, stream=None, tqdm_class=tqdm.tqdm):
        super().__init__(stream)
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa: E722
            self.handleError(record)


def _format_time(fmt, t, nsecs=None, precision=6):
    if "%f" in fmt:
        if nsecs is None:
            if isinstance(t, float):
                nsecs = int((t - int(t)) * 1e6)
            else:
                nsecs = 0
        if precision < 6:
            nsecs = int(nsecs * (0.1 ** (6 - precision)))
        fmt = fmt.replace("%f", "{:0{prec}d}".format(nsecs, prec=precision))
    return time.strftime(fmt, t)


class Formatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = _format_time(datefmt, ct, int(record.msecs * 1000), 3)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s


class ColoredFormatter(Formatter):
    class Color(int, Enum):
        BLACK = 30
        RED = 31
        GREEN = 32
        YELLOW = 33
        BLUE = 34
        MAGENTA = 35
        CYAN = 36
        WHITE = 37

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[%dm"
    FORMAT = COLOR_SEQ + "%s" + RESET_SEQ
    COLORS = {
        logging.CRITICAL: Color.RED,
        logging.FATAL: Color.RED,
        logging.ERROR: Color.RED,
        logging.WARNING: Color.YELLOW,
        logging.WARN: Color.YELLOW,
        logging.INFO: Color.WHITE,
        logging.DEBUG: Color.CYAN,
        logging.NOTSET: None,
    }

    def format(self, record):
        s = super().format(record)
        color = self.COLORS.get(record.levelno)
        if color is not None:
            s = self.FORMAT % (color, s)
        return s
