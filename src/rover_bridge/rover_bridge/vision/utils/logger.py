"""Simple logger wrapper — consistent format across all modules."""
import logging
import os

def RoverLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)-20s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
