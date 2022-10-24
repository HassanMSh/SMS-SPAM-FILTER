import pandas as pd

df = pd.read_csv('Datasets/edrs_2022.10.14_000000_TASK3067114.csv',
                 sep=';', header=0, on_bad_lines='skip')

to_drop = ['Client product',
           'Country',
           'UDH',
           'Vendor product',
           'Client MCCMNC (net)']

df.drop(to_drop, inplace=True, axis=1)
df.replace(regex=r'\W\s', value='', inplace=True)
df.replace(regex=r'\d', value='', inplace=True)
df.replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
df.drop_duplicates(inplace=True)
df.to_csv('Datasets/NewData.csv')
