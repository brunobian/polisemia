import pandas as pd

df = pd.read_csv('Stimuli.csv')
print(df.head())
first_row = df.iloc[32]

latex_table = first_row.to_latex()

print(latex_table)

