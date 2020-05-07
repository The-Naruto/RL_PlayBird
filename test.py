import pandas as pd
import numpy as np

pd1 = pd.DataFrame(np.arange(20).reshape((4,5)))

# pd1.to_csv('aa.csv')


Memory = 'mem.csv'


print(pd.Series([0]*2,index=[0,1],name=(111),))


q_table = pd.read_csv(Memory,index_col=0,header=[0,1])






