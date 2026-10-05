import logging
import sys

from tqdm import tqdm

# thin-line progress bar; the bar itself has a fixed width, so a long
# description makes the line longer instead of squeezing the bar
PBAR_STYLE = dict(
    ascii=" ─",
    bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
)


class TqdmLoggingHandler(logging.Handler):
    """
    print log records through tqdm, so a message arriving while a progress bar
    is active lands on its own line above the bar instead of on the bar's line
    """

    def emit(self, record):
        tqdm.write(self.format(record), file=sys.stderr)


def setup_logging(level=logging.INFO):
    """
    route the root logger through tqdm, in place of logging.basicConfig

    level : root logging level
    """
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])


def pbar(iterable, **kwargs):
    """
    tqdm progress bar in the shared thin-line style

    iterable : what to iterate over
    kwargs   : extra tqdm arguments, overriding the style
    out      : tqdm iterator
    """
    return tqdm(iterable, **{**PBAR_STYLE, **kwargs})
