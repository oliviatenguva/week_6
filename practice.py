#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# %%
#load data, column titles in the first row
salary_data = pd.read_csv('2025_salaries.csv', header=1, encoding='latin-1')
stats = pd.read_csv('nba_2025.txt', sep = ",", encoding='latin-1')
# %%
salary_data.head()
stats.head()
# %%
merged_data = pd.merge(salary_data, stats, on='Player')
# %%
merged_data.head()
# %%
duplicates = merged_data[merged_data.duplicated(subset='Player', keep=False)]
print(duplicates)

# %%


# %%
#Sklearn 4Steps:
#1: Create an instance of the model: mymodel = KMeans(n_clusters=3) : creating the model
#2: Fit the model to the data example: mymodel.fit(X) : training data
#3: Make predictions using the model: example: predictions = mymodel.predict(Y) 
#4: Evaluate the model's performance. example: score = mymodel.score(X)

#For kmeans, you don't need to predict, you can just use the labels_ attribute
#to get the cluster assignments for each data point after fitting the model.

#Lamda functions are anonymous functions used to create a new column in a dataframe. 
#Ex: creating a new salary in thousands column
merged_data['Salary_in_thousands'] = merged_data['Salary'].apply(lamda x: x/1000)
merged_data['High_Salary'] = merged_data['Salary'].apply(lamda x: True if x > 100000)

#Should be making a scatterplot with the best two features.
#Will have 3 clusters across 2 different variables. Pick which players will be the best under different requirements. 
#Good players will have the best statistics for the best price (lower salary). Include all the stats in your clusters. 
#We want players in the middle cluster as they will be better not to be overpaid and they are not bad at basketball.
#We want something in between. 
#Minutes played, Rebounds, Points
#Reduce the space: just do averages or do the nimutes. do not do both.
#Want to select the variable that has the alrgest variance. 
#Correlation: tracks the direction and velocity of data in space.
#Put data on the chart to see what variables are highly correlated wtih salary. 


#the best 2 features to predict. salary should show up as a heat map (salary is the color). Salary is not on the x and y axis. it is only used as a color.

# %%
