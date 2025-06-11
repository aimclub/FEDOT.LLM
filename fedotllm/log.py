import logging

logger = logging.getLogger("FEDOTLLM")
logger.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG to allow debug messages

file_handler = logging.FileHandler("fedotllm.log", mode="w")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

if __name__ == "__main__":
    # Example usage:
    logger.info("This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print("Log messages written to fedotllm.log")
