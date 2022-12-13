import os
import logging
import logging.config

from src.io.path_definition import get_project_dir

logging.config.fileConfig(os.path.join(get_project_dir(), 'config', 'logging.conf'))
logger = logging.getLogger('MainLogger')
logger.setLevel(logging.DEBUG)