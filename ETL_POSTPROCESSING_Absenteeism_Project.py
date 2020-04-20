# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from absenteeism_module import *
model = absenteeism_model('model', 'scaler')

model.load_and_clean_data('Absenteeism_new_data.csv')

model.predicted_outputs()
# -

import pymysql

conn = pymysql.connect(database = 'predicted_outputs', user = 'nativeuser', password = '365Pass')

cursor = conn.cursor()

# # # Checkpoint 'df_new_obs'

df_new_obs = model.predicted_outputs()
df_new_obs

# ## .execute()

cursor.execute('SELECT * FROM predicted_outputs;')

# ## Creating the INSERT Statement

insert_query = 'INSERT INTO predicted_outputs VALUES '

insert_query

df_new_obs.shape

df_new_obs['Age']

df_new_obs[df_new_obs.columns.values[6]][0]

for i in range(df_new_obs.shape[0]):
    insert_query += '(' 
    
    for j in range(df_new_obs.shape[1]):
        insert_query += str(df_new_obs[df_new_obs.columns.values[j]][i]) + ', '
    
    insert_query = insert_query[:-2] + '), '  

insert_query

insert_query = insert_query[:-2] + ';'

insert_query

cursor.execute(insert_query)

conn.commit()

conn.close()
