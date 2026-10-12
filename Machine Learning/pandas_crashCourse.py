
import sys

from pandas.conftest import axis_1

print(">>> Python executable:", sys.executable)

import pandas as pd

#Series
values = [10, 20, 30, 40, 50]
s = pd.Series(values, index=['a', 'b', 'c', 'd', 'e'])
print(s.loc['a'])

print(s)


# 2. Data Frame
df = pd.DataFrame({
    'name': ['Mike', 'Bob'],
    'age': [20, 40],
})
df = df.set_index('name') #name become index column
#always set df = df.set_index --> df.set_index just return a data frame
#where index is name, the actual change is not applied to data frame or df.set_index('name', inplace=True) [not recommend]

df.loc['Mike'] #can do that cause name change to index

df1 = pd.DataFrame({
    'a': [1, 2, 3]
}, index=[0,1,2])
df2 = pd.DataFrame({
    'a': [10, 20, 30]
}, index=[1,2,0])
print(df1 + df2) # when do math, it's aligned based on the index

df.reset_index() #reset the index --> return index to an actual column

df.to_csv('mydata.csv', index=None) #change to other format
# index=None --> get rid of index while exporting the file (import without index) --> cause 'unnamed zero' bug

pd.read_csv('mydata.csv', index_col=0) # pandas recognizing the column unnamed --> 'unnamed zero' --> call unnamed zero
# --> index_col=0 specify first column of the csv is actually the index column

df.to_json('')

df.to_dict()


from sklearn.datasets import fetch_california_housing

df2 = fetch_california_housing(as_frame=True).frame
df2.head()
df2.tail(10) # get the last 10 rows, default 5
df2.sample(10) # get 10 random row, default 1

list(df2.columns) # get all columns

pd.options.display.max_columns = 5 # set to see 5 columns

df2.info() #give information about the data frame

df2.describe() #summarize all the statistics: count, mean, std, min, 25%, 50%, 75%, max

type(df2.HouseAge) #df.columnName --> column of data frame is series
#or
df2['HouseAge']

df2['HouseAge'].mean() # max(), min(), std(), median(), mode(), count(), sum()

import matplotlib.pyplot as plt
df2['HouseAge'].hist(figsize=(12, 8)) # get histogram using matplotlib
plt.tight_layout()

df2['HouseAge'].plot(figsize=(12, 8)) # produce a line chart

plt.title('House Age')
plt.show()

import yfinance as yf
stock_df = yf.download('AAPL')
stock_df.plot()
stock_df.hist()


#ACCESSING DATA
df3 = pd.DataFrame({
    'name': ['Mike', 'Alice'],
    'age': [30, 45],
    'job': ['dev', 'designer']
})

df3.loc[1] #take sample in index = 1
df3.iloc[1]# an integer indexing attribute

df3 = df3.set_index('name')
df3.loc['Alice', 'age'] # get the specific field row Alice, column age
#or
df3.iloc[1, 0]

df3.at['Alice', 'age'] #df3.at['Alice'] didn't work, cause it get 1 specific field/value
df3.iat[1, 0]

df3.at['Alice', 'age'] = 60 # change field has age column and Alice row
df3.loc['Alice'] = [75, 'engineer'] #Change row Alice

df3.loc['John'] = [20, 'teacher'] #append new row

df3.iloc[0:2] #slicing
df3.iloc[0:3, 1] #can do for individual axis, 1 means second column
df3.iloc[:, 1] # give all of sample but only column 1
df3.iloc[:, :] #take all

df3.age = df3.age * 2 #do math individual cells , /

def myFunction(x):
    if x % 3 == 0:
        return x ** 2
    else:
        return x // 2

df3 = df3.age.apply(myFunction) #apply function
df3.age.apply(lambda x: x ** 2 if x % 10 == 0 else x // 2)

def myfunction2(x):
    if x.endswith('r'):
        return 'Without Job'
    else:
        return x
df3 = df3.job.apply(myfunction2)

df3['summary'] = df3.apply(lambda row: f'Age: {row['age']}, Job: {row['job']}', axis=1)
df3.drop('summary', axis=1)

