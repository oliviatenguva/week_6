# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
help(pd.read_csv)
df = pd.read_csv('house_votes_Dem.csv', encoding='latin-1')

# %%
# take a look at the data
df.head()
df.info()
# %%
# separate out the numeric features
c_num = df[['handicapped-infants', 'water-project-cost-sharing']]
# %%
# documentation for kmeans in sklearn
help(kMeans)

# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(c_num)

# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_
# %%
  
# %% simple plot of the clusters
help(plt.scatter)
 
# %%
#Use a for loop to check different cluster numbers and see how the intertia changes
intertias= []
k_values = range(1, 10)
for k in k_values:
    k_means = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(c_num)
    intertias.append(k_means.inertia_)