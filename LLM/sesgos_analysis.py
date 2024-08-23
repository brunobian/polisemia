
import pandas as pd

# Set the working directory and read the CSV files
# os.chdir("~/Desktop/Tesis/polisemia/LLM/")  # Uncomment and modify as needed if you need to change directory
df = pd.read_csv("distancias.csv", sep=",")
df_w = pd.read_csv("distancias_nuevo_modelo.csv", sep=",")

# Add a new column to each dataframe
df['Model'] = 'GPT2 base'
df_w['Model'] = 'GPT2 WordLevel'

# Combine the dataframes
df = pd.concat([df_w, df], ignore_index=True)

# Convert columns to numeric
df['baseS1'] = pd.to_numeric(df['2'], errors='coerce')
df['SignS1'] = pd.to_numeric(df['3'], errors='coerce')
df['baseS2'] = pd.to_numeric(df['5'], errors='coerce')
df['SignS2'] = pd.to_numeric(df['6'], errors='coerce')

# Calculate differences and normalize
df['diff_base1_emb'] = (df['SignS1'] - df['baseS1']) / abs(df['baseS1'])
df['diff_base2_emb'] = (df['SignS2'] - df['baseS2']) / abs(df['baseS2'])

# Read and prepare the second CSV file
df2 = pd.read_csv("../comportamental/accuracies.csv", sep=",")
df2 = df2[['indTarget', 'diff_base1', 'diff_base2']]

# Convert columns to numeric
df2['diff_base1'] = pd.to_numeric(df2['diff_base1'], errors='coerce')
df2['diff_base2'] = pd.to_numeric(df2['diff_base2'], errors='coerce')

# Merge the dataframes based on the specified columns
df = pd.merge(df, df2, left_on='0', right_on='indTarget')

# Save the dataframe to a CSV file if needed
df.to_csv("processed_data.csv", index=False)


