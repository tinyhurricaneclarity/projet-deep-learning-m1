"""heatmap pour comparer les paramètres"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/autofs/unityaccount/cremi/leanguye/projet-deep-learning-m1/resnet18/src/resnet18_RGB/results/grid_search_results.csv")

#pour séparer les paramètre dans la colonne config
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.duplicated()].copy()

df['optimizer'] = df['optimizer'].astype(str)
df['scheduler'] = df['scheduler'].astype(str)

# extraction des hyperparamètres depuis config
extracted = df['config'].str.extract(
    r'lr(?P<lr>[\d.]+)_opt(?P<optimizer>\w+)_sched(?P<scheduler>\w+)_step(?P<step>\d+)_gamma(?P<gamma>[\d.]+)'
)

df = pd.concat([df, extracted], axis=1)

# conversions de types
df['lr'] = df['lr'].astype(float)
df['gamma'] = df['gamma'].astype(float)
df['step'] = df['step'].astype(int)



# exemple : on agrège la performance
pivot = df.pivot_table(
    values="val_acc",   # <-- à adapter
    index="lr",
    columns="step",
    aggfunc="mean"
)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title("Heatmap accuracy (lr vs step)")
plt.show()