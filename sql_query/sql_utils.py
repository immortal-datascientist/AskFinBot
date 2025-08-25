
############################## original code for Mysql db ###############################################
# # sql_utils.py
# import pymysql
# import pandas as pd


# def connect_to_db(
#     host="localhost",
#     user="root",
#     password="immortal@123",
#     database="employee",
#     port=3306
# ):
#     return pymysql.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database,
#         port=port
#     )

# def execute_sql_query(sql_code: str):
#     try:
#         conn = connect_to_db()
#         df = pd.read_sql(sql_code, conn)
#         conn.close()
#         return df, None
#     except Exception as e:
#         return None, str(e)

###################################################################################################

# sql_utils.py
import oracledb
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- DB CONNECTION CONFIG ---
ORACLE_CONFIG = {
    "user": "custom",
    "password": "custom",
    "host": "192.168.1.150",
    "port": 1521,
    "service_name": "IFIT"
}

def connect_to_oracle():
    return oracledb.connect(**ORACLE_CONFIG)

def get_all_users():
    conn = connect_to_oracle()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM all_users ORDER BY username")
    users = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return users

def get_user_tables(username):
    conn = connect_to_oracle()
    cursor = conn.cursor()
    cursor.execute(f"SELECT table_name FROM all_tables WHERE owner = UPPER(:username)", {"username": username})
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return tables

def fetch_table_preview(username, table_name, row_limit=5):
    conn = connect_to_oracle()
    query = f'SELECT * FROM "{username}"."{table_name}" FETCH FIRST {row_limit} ROWS ONLY'
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def execute_sql_query(sql_code: str):
    try:
        conn = connect_to_oracle()
        df = pd.read_sql(sql_code, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)



