import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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












# Having used the commands above to see the breakdown of the data, it can be seen that there is a large variance between the mean unit price (4.611) and the mean customer ID (15287).
# This would create problems with the algorithm as it would mean there would be a bias towards the larger value. Imagine looking at a barchart where the scale is thrown out due to a large
# value. It would make it very difficult to read the smaller value. Therefore, scaling is needed to align the 2 values to allow for correct analysis

# First we instantiate the scaler