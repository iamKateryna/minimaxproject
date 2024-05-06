from datetime import datetime
import logging

def setup_logger(filename):
    # Configure logger
    logging.basicConfig(filename=filename,  filemode='w', level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Started Logging")


