'''
Created on Dec 7, 2017

@author: ghazy
'''
import logging

### Currently not using this class anywhere, moved this code to __init__.py
class SBLogger(object):

    def __init__(self,
                 log_file_name,
                 console_logging_level,
                 file_logging_level):
        
        cll = None
        fll = None
        logging_levels_dict = { 'WARING': logging.WARNING,
                                'DEBUG': logging.DEBUG,
                                'CRTICAL': logging.CRITICAL,
                                'ERROR': logging.ERROR,
                                'INFO': logging.INFO }
        try:
            cll = logging_levels_dict[console_logging_level]
            fll = logging_levels_dict[file_logging_level]
        except KeyError:
            # default to warning level
            cll = logging.DEBUG
            fll = logging.DEBUG
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(cll)
        logger_handler = logging.FileHandler(log_file_name)
        logger_handler.setLevel(fll)
        
        logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger_formatter.setFormatter(logger_formatter)
        
        self.logger.addHandler(logger_handler)
        self.logger.info('Completed configuring logger()!')
        
        
        