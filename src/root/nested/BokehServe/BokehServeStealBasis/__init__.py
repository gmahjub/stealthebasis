from os import path, remove
import logging
import logging.config
import json


def delete_existing_log_file(log_file_name):
    if path.isfile(log_file_name):
        remove(log_file_name)


def create_logger_from_json(json_log_config_file):
    with open(json_log_config_file, 'r') as logging_config_file:
        config_dict = json.load(logging_config_file)
    logging.config.dictConfig(config_dict)


def create_logger(log_file_name,
                  console_logging_level,
                  file_logging_level):
    cll = None
    fll = None
    logging_levels_dict = {'WARNING': logging.WARNING,
                           'DEBUG': logging.DEBUG,
                           'CRTICAL': logging.CRITICAL,
                           'ERROR': logging.ERROR,
                           'INFO': logging.INFO}
    try:
        cll = logging_levels_dict[console_logging_level]
        fll = logging_levels_dict[file_logging_level]
    except KeyError:
        cll = logging.DEBUG
        fll = logging.DEBUG

    console = logging.StreamHandler()
    console.setLevel(cll)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(console_formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(cll)
    logger_handler = logging.FileHandler(log_file_name)
    logger_handler.setLevel(fll)
    logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.addHandler(console)

    logger.info('Completed configuring logger()!')
    logger.info('this is being run, idiot!')


def get_logger():
    return logging.getLogger(__name__)


def main():
    create_logger('stealthebasis.log',
                  'DEBUG',
                  'DEBUG')
    logger = get_logger()
    logger.info("nested.init.Main(): Instantiated a logger...")


main()
