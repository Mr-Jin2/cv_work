import os
import sys
from loguru import logger

def setup_logger(save_dir,
                 filename='runtime.log',
                 mode='a',
                 level='DEBUG'):
    '''
    Return:
        logger instance
    '''
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == 'o' and os.path.exists(save_file):
        os.remove(save_file)

    logger.add(
        sys.stderr,
        format=loguru_format,
        level=level,
        enqueue=True
    )
    logger.add(save_file)
