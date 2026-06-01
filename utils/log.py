import logging
from pathlib import Path

from rich.logging import RichHandler

_ROOT_LOGGER = logging.getLogger("isgsa")
# Set logger to debug and then filter more using handlers below
_ROOT_LOGGER.setLevel(logging.DEBUG)

console_handler = RichHandler()
console_handler.setLevel(logging.INFO)
_ROOT_LOGGER.addHandler(console_handler)

# Absolute path so worker processes (which chdir into workspace/) can still find it.
_LOG_PATH = Path(__file__).resolve().parent.parent / "out" / "debug.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(str(_LOG_PATH))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="[%Y/%m/%d %H:%M:%S]",
    )
)

_ROOT_LOGGER.addHandler(file_handler)


def set_log_level(level: str):
    level = level.upper()
    if level in logging._nameToLevel:
        console_handler.setLevel(level)
        _ROOT_LOGGER.debug("LOG LEVEL IS SET TO DEBUG")
    else:
        raise ValueError(f"Unknown log level: {level}")
