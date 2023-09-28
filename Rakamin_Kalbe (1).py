#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np


# In[59]:


dt = pd.read_csv('Transaction.csv', delimiter=';')
dc = pd.read_csv('Customer.csv', delimiter=';')
dp = pd.read_csv('Product.csv', delimiter=';')
ds = pd.read_csv('Store.csv', delimiter=';')


# In[60]:


dt = dt.dropna()
dc = dc.dropna()
dp = dp.dropna()
ds = ds.dropna()


# In[61]:


dt


# In[62]:


dc


# In[63]:


dp


# In[64]:


ds


# In[65]:


print(dt.columns)
print(dc.columns)
print(dp.columns)
print(ds.columns)


# In[66]:


#mergerdata

merged_data = pd.merge(dt, dc, on='CustomerID')
merged_data


# In[67]:


merged_data = pd.merge(merged_data, dp, on='ProductID')
merged_data = pd.merge(merged_data, ds, on='StoreID')
merged_data


# In[68]:


# Data baru untuk analisis regresi time series

daily_sales = merged_data.groupby('Date')['Qty'].sum().reset_index()

daily_sales


# In[69]:


daily_sales.index.freq = 'D'  # Frekuensi harian

# Cek stasioneritas data
def check_stationarity(ts):
    # Hitung rolling statistics
    rolling_mean = ts.rolling(window=30).mean()
    rolling_std = ts.rolling(window=30).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original Data')
    plt.plot(rolling_mean, label='Rolling Mean (30 days)')
    plt.plot(rolling_std, label='Rolling Std (30 days)')
    plt.legend()
    plt.title('Rolling Statistics')
    plt.show()

# Cek stasioneritas data daily_sales
check_stationarity(daily_sales['Qty'])


# In[70]:


# Mengurangkan rolling mean dari data untuk membuat data lebih stasioner
daily_sales['Qty_diff'] = daily_sales['Qty'] - daily_sales['Qty'].rolling(window=30).mean()

# Cek stasioneritas data yang sudah di-differencing
check_stationarity(daily_sales['Qty_diff'].dropna())


# In[71]:


# Plot ACF dan PACF
plot_acf(daily_sales['Qty_diff'].dropna(), lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(daily_sales['Qty_diff'].dropna(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# In[74]:


from statsmodels.tsa.arima.model import ARIMA

p = 1
d = 1  
q = 1

model = ARIMA(daily_sales['Qty'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())


# In[75]:


#validasi model
# Memisahkan data menjadi data pelatihan dan data uji
train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales[:train_size], daily_sales[train_size:]


# In[76]:


# Membuat model ARIMA dengan nilai p, d, dan q yang sesuai
model = ARIMA(train['Qty'], order=(p, d, q))
model_fit = model.fit()


# In[78]:


# Prediksi dengan model yang telah dilatih
forecast = model_fit.forecast(steps=len(test))

# Tampilkan hasil prediksi
print(f'Prediksi total kuantitas harian produk:')
print(forecast)


# In[79]:


# Hitung metrik evaluasi (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['Qty'], forecast)
print(f'Mean Squared Error (MSE): {mse}')


# In[81]:


#Prediksi Total Kuantitas Harian Produk
# Prediksi total kuantitas harian produk untuk 7 hari ke depan
forecast_days = 7
forecast_future = model_fit.forecast(steps=forecast_days)

# Tampilkan hasil prediksi
print(f'Prediksi total kuantitas harian produk untuk {forecast_days} hari ke depan:')
print(forecast_future)



# In[83]:


#Membuat Data Baru untuk Clustering
cluster_data = merged_data.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()

cluster_data


# In[85]:


from sklearn.preprocessing import StandardScaler

# Kolom yang akan di-standarisasi (fitur-fitur untuk clustering)
features = ['TransactionID', 'Qty', 'TotalAmount']

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Standarisasi data pada kolom yang dipilih
cluster_data[features] = scaler.fit_transform(cluster_data[features])

cluster_data


# In[100]:


from sklearn.cluster import KMeans

# Menghapus fitur 'Cluster' dari data latihan
cluster_data = cluster_data.drop(columns=['Cluster'])

# Pilih jumlah cluster (K)
k = 3

# Inisialisasi model K-Means
kmeans = KMeans(n_clusters=k)

# Melatih model
kmeans.fit(cluster_data)


cluster_data


# In[107]:


from sklearn.cluster import KMeans

# Pilih jumlah cluster (K)
k = 3

# Inisialisasi model K-Means
kmeans = KMeans(n_clusters=k)

# Melatih model
kmeans.fit(cluster_data)

# Menambahkan label kluster ke data
cluster_data['Cluster'] = kmeans.labels_

# Analisis hasil
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Menampilkan hasil
print("Centroids:")
print(cluster_centers)
print("Labels:")
print(cluster_labels)


# In[112]:


data = pd.DataFrame({
    'CustomerID': range(1, 445),
    'TransactionID': range(1, 445),  # Panjang yang sama dengan 'CustomerID'
    'Qty': [10] * 444,  # Panjang yang sama dengan 'CustomerID'
    'TotalAmount': [1000] * 444,  # Panjang yang sama dengan 'CustomerID'
    'Cluster': [0] * 444  # Panjang yang sama dengan 'CustomerID'
})

cluster_stats = data.groupby('Cluster').agg({
    'TransactionID': ['count', 'mean', 'std'],
    'Qty': ['mean', 'median', 'std'],
    'TotalAmount': ['mean', 'median', 'std']
}).reset_index()

print(cluster_stats)


# In[113]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi hasil klustering dengan plot scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_data, x='TransactionID', y='Qty', hue='Cluster', palette='Set1', s=100)
plt.title('Visualisasi Hasil Klustering')
plt.xlabel('TransactionID')
plt.ylabel('Qty')
plt.legend(title='Cluster')
plt.show()


# In[ ]:





# In[ ]:




