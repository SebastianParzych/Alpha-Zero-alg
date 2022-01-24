"""Shared logging between modules"""
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
console_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)-8s %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logging.basicConfig()
logger.setLevel(logging.INFO)