import sys
from loguru import logger

def setup_logger():
    logger.remove()
    logger.add(sys.stdout, 
            level="INFO",
            format="[{time:HH:mm:ss}]<level>{message}</level>")

    logger.add("logs/agents.log", 
            level="DEBUG",
            format="{time:HH:mm:ss} | <level>{level}</level> |\n<level>{message}</level>",
            mode="w")

def get_logger():
    return logger