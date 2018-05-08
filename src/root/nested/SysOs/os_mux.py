'''
Created on Dec 4, 2017

@author: ghazy
'''
import platform
import getpass
import os
from root.nested import get_logger

class OSMuxImpl(object):
    
    def __init__(self):
        
        logger = get_logger()
        self.system = platform.system()
        self.user = getpass.getuser()
        
        if self.system == 'Windows':
            self.home_dir = os.environ['HOMEPATH']
        else:
            self.home_dir = os.environ['HOME']
        logger.info('OSMuxImpl.__init__.home_dir: ' + self.home_dir)
    
    @staticmethod  
    def get_program_user_username():
        
        return getpass.getuser()
    
    @staticmethod
    def get_system():
        
        return platform.system()
    
    @staticmethod
    def get_home_dir():
        
        if OSMuxImpl.get_system() == 'Windows':
            return os.environ['HOMEPATH']
        else:
            return os.environ['HOME']
    
    @staticmethod
    def get_dir_div():
        
        if OSMuxImpl.get_system() == 'Windows':
            return '\\'
        else:
            return '/'
    
    @staticmethod
    def get_proper_path(user_provided_path):
        
        try:
            path_tokens = user_provided_path.split('/')    
        except Exception as excp:
            print ('unexpected exception thrown', excp)
            return None
        
        if OSMuxImpl.get_system() == 'Windows':
            return_path = "C:" + OSMuxImpl.get_home_dir()
        else:
            return_path = OSMuxImpl.get_home_dir()
        
        for token in path_tokens:
            if token == '':
                continue
            return_path += (OSMuxImpl.get_dir_div() + token)
        return_path += OSMuxImpl.get_dir_div()
        return return_path
        
        
            
        
        
            
        
        
        