import pandas as pd

df = pd.read_csv('Datasets/NewData.csv')
df.drop(['x'], inplace=True, axis=1)
df.dropna(inplace=True)

df = df[df['SMStext'].map(lambda x: x.isascii())]

df.to_csv('Datasets/NewDataASCII.csv')
