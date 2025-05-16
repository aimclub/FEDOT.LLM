import logging

logger = logging.getLogger("FEDOTLLM")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

if __name__ == "__main__":
    # Example usage:
    logger.info("This is an informational message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    print("Log messages written to app.log")
