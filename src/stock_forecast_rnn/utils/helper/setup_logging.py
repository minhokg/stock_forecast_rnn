import logging
import sys
from pathlib import Path

import colorlog


def setup_logging(save_path: str = str(Path.cwd())) -> None:
    """
    Set up logging.

    :param save_path: Defines where to save logs.
    :return: None
    """
    # Suppress Matplotlib font manager debug messages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Make sure we catch all uncaught exceptions
    sys.excepthook = _excepthook

    # Ensure save_path exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # define a colored formatter for both file and console
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)-8s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # create a file handler that uses the colored formatter.
    # Note: The file will contain ANSI escape codes.
    file_handler = logging.FileHandler(filename=Path(save_path) / "app.log", mode="w")
    file_handler.setFormatter(formatter)

    # create a stream handler that also uses the colored formatter.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # configure the root logger with both handlers.
    # we omit the format parameter because each handler has its own formatter.
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
    )

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def _excepthook(*args):  # noqa: ANN002
    """
    Catch all uncaught exceptions and log them.

    :param args: Variable number of positional arguments.
    :return: None.
    """
    logging.getLogger().error("Uncaught exception:", exc_info=args)
