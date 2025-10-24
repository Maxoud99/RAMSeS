from pathlib import Path
import os
from loguru import logger
BASE_DIR = Path(__file__).resolve().parent.parent
path = os.path.join(BASE_DIR, '')
db_path = os.path.join(path, 'db/tsad.db')
logger.info(f'{__file__}\ncurrent base dir is path is {path}\ndb path is {db_path}')

import sqlite3 as sl

def get_db_conn():
    return sl.connect(db_path,check_same_thread=False)

# get_db_conn()