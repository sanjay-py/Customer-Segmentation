#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler
import plotly.express as px


# In[2]:


customer_data = pd.read_csv("Mall_Customers.csv")


# In[5]:


# Data exploration
print(customer_data.info())
print(customer_data.columns)
print(customer_data.head())


# In[6]:


print(customer_data['Age'].describe())
print("SD of Age:", customer_data['Age'].std())


# In[9]:


print(customer_data['Annual Income (k$)'].describe())
print("SD of Annual Income:", customer_data['Annual Income (k$)'].std())


# In[10]:


print(customer_data['Spending Score (1-100)'].describe())
print("SD of Spending Score:", customer_data['Spending Score (1-100)'].std())


# In[12]:


# Gender Visualization
gender_counts = customer_data['Gender'].value_counts()


# In[13]:


# Bar Plot
gender_counts.plot(kind='bar', color=['orange', 'skyblue'])
plt.title("Gender Comparison")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In[14]:


# Pie Chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title("Gender Distribution")
plt.show()


# In[15]:


# Age Distribution
# Histogram
plt.hist(customer_data['Age'], color='blue', bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[16]:


# Boxplot
sns.boxplot(x=customer_data['Age'], color='magenta')
plt.title("Boxplot of Age")
plt.show()


# In[17]:


# Annual Income Distribution
plt.hist(customer_data['Annual Income (k$)'], bins=10, color='maroon')
plt.title("Annual Income Histogram")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Frequency")
plt.show()


# In[18]:


# Density plot
sns.kdeplot(customer_data['Annual Income (k$)'], shade=True, color="yellow")
plt.title("Annual Income Density Plot")
plt.show()


# In[19]:


# Boxplot of Spending Score
sns.boxplot(x=customer_data['Spending Score (1-100)'], color='darkred')
plt.title("Boxplot of Spending Score")
plt.show()


# In[20]:


# Histogram of Spending Score
plt.hist(customer_data['Spending Score (1-100)'], bins=10, color='purple')
plt.title("Spending Score Histogram")
plt.xlabel("Spending Score")
plt.ylabel("Frequency")
plt.show()


# In[21]:


# K-Means Clustering - Elbow Method
X = customer_data.iloc[:, [2, 3, 4]]


# In[22]:


# Elbow Plot
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(X)
visualizer.show()


# In[31]:


# Silhouette Analysis
for k in range(2, 5):
    kmeans = KMeans(n_clusters=k, random_state=123)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"For k={k}, Silhouette Score={score:.4f}")
    viz = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    viz.fit(X)
    viz.show()


# In[24]:


# Final KMeans Model with 6 Clusters
kmeans = KMeans(n_clusters=6, random_state=125)
customer_data['Cluster'] = kmeans.fit_predict(X)


# In[25]:


# Print centroids
print("Cluster Centers:\n", kmeans.cluster_centers_)


# In[27]:


# PCA and Visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)


# In[28]:


plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:,0], pca_components[:,1], c=customer_data['Cluster'], cmap='rainbow')
plt.title("KMeans Clustering using PCA Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()


# In[29]:


# Cluster Visualization using Plotly
fig = px.scatter(customer_data, x='Annual Income (k$)', y='Spending Score (1-100)',
                 color=customer_data['Cluster'].astype(str),
                 title='Customer Segments (Annual Income vs Spending Score)',
                 labels={'color': 'Cluster'})
fig.show()


# In[30]:


fig2 = px.scatter(customer_data, x='Spending Score (1-100)', y='Age',
                 color=customer_data['Cluster'].astype(str),
                 title='Customer Segments (Spending Score vs Age)',
                 labels={'color': 'Cluster'})
fig2.show()


# In[ ]:




