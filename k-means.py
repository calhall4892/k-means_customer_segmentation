import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_excel(r'C:\Users\Callum\Documents\ML Researcher\k-means_customer\k-means_customer_segmentation\data\Online Retail.xlsx')
print(df.head())
print(df.describe())
print(df.info())

# I can see that there are a number of NaN values which need to be cleaned up as well as negative mins.
# This line drops all of the nan values from CustomerID
df = df.dropna(subset=['CustomerID'])

# This one removes all the rows that show Quantity as a negative
df[df['Quantity'] > 0]

# Feature Engineering - We can use a technique called RFM analysis. this is recency, frequency and monitary
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

rfm_df = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'nunique',
    'InvoiceDate': 'max'
})

print(rfm_df.head())

# Now after the feature engineering, we can tidy up the new df and rename columns to better names.
rfm_df = rfm_df.rename(columns={
    'TotalPrice': 'Monetary Spend',
    'InvoiceNo': 'Frequency',
    'InvoiceDate': 'Recency'
})

snapshot_date = df['InvoiceDate'].max()
rfm_df['Recency'] = (snapshot_date - rfm_df['Recency']).dt.days


# Having used the commands above to see the breakdown of the data, it can be seen that there is a large variance between the mean unit price (4.611) and the mean customer ID (15287).
# This would create problems with the algorithm as it would mean there would be a bias towards the larger value. Imagine looking at a barchart where the scale is thrown out due to a large
# value. It would make it very difficult to read the smaller value. Therefore, scaling is needed to align the 2 values to allow for correct analysis

# First we instantiate the scaler
scaler = StandardScaler()

# .fit() is the learning step. The scaler analyzes your data to learn its properties. For StandardScaler, this means calculating the mean and standard deviation for each column.
#  It doesn't change the data itself.
# .transform() is the applying step. It uses the parameters learned during the .fit() step to actually scale the data. You can't transform data that hasn't been fitted first.
# There is a single method to use both and its .fit_transform()
rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary Spend']])

# The scaler outputs as a NumPy array rather than dataframe so we convert it back to a df
rfm_df_scaled = pd.DataFrame(data=rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

#-------------------------------------------------------------------------------
# K-means clustering

# To find out the optimum number of clusters, we can run a number of different methods including the elbow method. This is where the model runs multiple times against each k number
# We then calculate a number called the inertia which is how tightly packed the clusters are. We then plot this in order to see which is the best k number for our use case.

# create an empty list to store values
inertia_scores = []

# define the k values to test
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
# now we fit and append the data into our list
    kmeans_fit = kmeans.fit(rfm_df_scaled)
    inertia_scores.append(kmeans_fit.inertia_)

# Plotting the elbow curve
plt.plot(k_values, inertia_scores)
plt.title('K numbers against Inertia')
plt.xlabel('K numbers')
plt.ylabel('Inertia')
plt.show()

# Now we can call the final model with the appropriate kvalues

kmeans_pred = KMeans(n_clusters=4, random_state=42).fit_predict(rfm_df_scaled)
rfm_df['Cluster'] = kmeans_pred

# Now we group by the cluster
cluster_summary = rfm_df.groupby('Cluster').mean()

# Make the plotting area larger
plt.figure(figsize=(15, 5))

# Plot for Recency
plt.subplot(1, 3, 1)
sns.barplot(x=cluster_summary.index, y=cluster_summary['Recency'])
plt.title('Average Recency')

# Plot of monetary
plt.subplot(1, 3, 2)
sns.barplot(x=cluster_summary.index, y=cluster_summary['Monetary Spend'])
plt.title('Average Monetary Spend')

# Plot for Frequency
plt.subplot(1, 3, 3)
sns.barplot(x=cluster_summary.index, y=cluster_summary['Frequency'])
plt.title('Average Frequency')

plt.tight_layout
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x=rfm_df['Recency'], y=rfm_df['Frequency'], z=rfm_df['Monetary Spend'], c=rfm_df['Cluster'])
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.show()