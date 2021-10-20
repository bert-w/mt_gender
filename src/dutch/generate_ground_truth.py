#%% 
import pandas as pd
from pathlib import Path
#%%
DATA_DIR = Path("data/aggregates")
df = pd.read_csv(DATA_DIR/'nl_manual.tsv', sep='\t', names=['gender', 'index', 'sentence', 'subject'])
df
#%%
indeces = []
errors = 0
for i, row in df.iterrows():
    tokens = row['sentence'].replace(',','').split() # kommas worden deel van woord
    try:
        index = tokens.index(row['subject'])
        indeces.append(index)
    except ValueError as e:
        print(e)
        print("row", i)
        print(row.values)
        errors+=1
        indeces.append(-1)
        

print(f"{errors=}")
# %%
df['index'] = indeces
df
# %%
df.to_csv(DATA_DIR/'nl.txt', sep='\t',header=None)
# %%
