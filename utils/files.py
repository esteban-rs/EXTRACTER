import math
import numpy as np
import logging
import cv2
import os
import shutil

class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        ### create a logger
        self.__logger = logging.getLogger(logger_name)

        ### set the log level
        self.__logger.setLevel(log_level)

        ### create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        ### create a handler to print on console
        console_handler = logging.StreamHandler()

        ### define the output format of handlers
        formatter = logging.Formatter('[%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        ### add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, 'model'))
    
    if ((args.eval and args.eval_save_results) or args.test):
        os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    
    for k, v in vars(args).items():
        args_file.write(k.rjust(30,' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name), 
        logger_name=args.logger_name).get_log()

    return _logger