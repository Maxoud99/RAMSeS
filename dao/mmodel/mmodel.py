from utils.db_utils import get_db_conn
from loguru import logger

con = get_db_conn()


def query_algorithm_name():
    sql = f"select ALGORITHM_NAME from UMS_TSAD_ALGORITHM"
    res = con.execute(sql).fetchall()
    algorithm_name_list = [i[0] for i in res]
    logger.info(f'algorithm_name is {algorithm_name_list}')
    return algorithm_name_list


_ = query_algorithm_name()


def insert_model_name(model_name):
    """Insert a model name into the UMS_TSAD_ALGORITHM table if it does not already exist."""

    check_query = ''' SELECT COUNT(*) FROM UMS_TSAD_ALGORITHM WHERE ALGORITHM_NAME = ?; '''
    cursor = con.cursor()
    cursor.execute(check_query, (model_name,))
    result = cursor.fetchone()
    if result[0] > 0:
        print("Model name already exists.")
        return False  # Indicate that the model was not inserted because it already exists

    # If the model name does not exist, insert it
    insert_query = ''' INSERT INTO UMS_TSAD_ALGORITHM(ALGORITHM_NAME) VALUES(?); '''
    cursor.execute(insert_query, (model_name,))
    con.commit()
    print("Model name inserted successfully.")
    return True  # Indicate successful insertion




def delete_algorithm_name(algorithm_name):
    """Delete an algorithm name from the UMS_TSAD_ALGORITHM table.

    Args:
        algorithm_name (str): Name of the algorithm to delete.
    """

    query = ''' DELETE FROM UMS_TSAD_ALGORITHM WHERE algorithm_name = ?; '''
    cursor = con.cursor()
    cursor.execute(query, (algorithm_name,))
    con.commit()
    if cursor.rowcount == 0:
        print("No such algorithm found to delete.")
        return False
    else:
        print("Algorithm deleted successfully.")
        return True