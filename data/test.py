import pandas as pd
from sqlalchemy import create_engine, exc
import sqlite3

# engine = create_engine('sqlite3:///DistasterResponse.db')
# c = engine.connect()
# # c.execute(text("SELECT 1"))

conn = sqlite3.connect('DisasterResponse.db')

# messages.to_sql('test', conn)
# c = conn.cursor()

# for row in c.execute('SELECT * FROM tbl_disaster_response'):
#     print(row)

# con.close()
df = pd.read_sql_query("SELECT * from tbl_disaster_response", conn)
print(df.head())
conn.close()